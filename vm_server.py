import os
import uuid
import logging
import multiprocessing
import threading
import base58
import re
import time
import signal
import base64
import bcrypt
import hashlib
import math
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import firestore
from google.cloud import run_v2
from solders.keypair import Keypair
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv

# 1. LOAD SECRETS
load_dotenv()

# --- DYNAMIC CORE ALLOCATION ---
# Calculate cores for grinding, reserving 1 for the API/System
TOTAL_CORES = multiprocessing.cpu_count()
TOTAL_GRINDING_CORES = max(1, TOTAL_CORES - 1) 

# 2. CONFIGURATION
PROJECT_ID = os.getenv('PROJECT_ID', 'vanityforge')
SOLANA_RPC_URL = os.getenv('SOLANA_RPC_URL')
TREASURY_PUBKEY = os.getenv('TREASURY_PUBKEY')
SMTP_EMAIL = os.getenv('SMTP_EMAIL')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
ADMIN_EMAIL = "tomaaytakin@gmail.com"

# --- QUEUE LIMITS (TRAFFIC CONTROL) ---
MAX_LOCAL_JOBS = 1    # Max concurrent "Easy" jobs (Uses TOTAL_GRINDING_CORES)
MAX_CLOUD_JOBS = 50  # Max concurrent "Hard" jobs on Cloud Run

# CLOUD JOB CONFIGURATION
GPU_JOB_NAME = f"projects/{PROJECT_ID}/locations/europe-west1/jobs/vanity-gpu-worker"

# 3. SAFETY CHECK
if not SOLANA_RPC_URL or not TREASURY_PUBKEY or not SMTP_PASSWORD:
    print("CRITICAL ERROR: Missing environment variables. Make sure .env exists.")
    exit(1)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": ["https://tomaaytakin.github.io", "https://vanityforge.org"]}})

@app.route('/')
def index(): return app.send_static_file('index.html')

@app.route('/roadmap')
def roadmap(): return app.send_static_file('roadmap.html')

@app.route('/faq')
def faq(): return app.send_static_file('faq.html')

def cleanup_firestore_client(db):
    try:
        if db:
            # Explicitly close the underlying gRPC channel
            db._channel.close()
    except Exception as e:
        print(f"Error during client cleanup: {e}")

def is_base58(s):
    if not s: return True
    return bool(re.match(r'^[1-9A-HJ-NP-Za-km-z]+$', s))

# --- DISPATCHER LOGIC (CLOUD RUN JOBS) ---
def dispatch_cloud_job(job_id, user_id, prefix, suffix, case_sensitive, pin):
    """Triggers the remote Cloud Run Job (8 vCPU or GPU)."""
    try:
        client = run_v2.JobsClient()
        request = run_v2.RunJobRequest(
            name=GPU_JOB_NAME,
            overrides={
                "container_overrides": [{
                    "env": [
                        {"name": "TASK_JOB_ID", "value": job_id},
                        {"name": "TASK_PREFIX", "value": prefix or ""},
                        {"name": "TASK_SUFFIX", "value": suffix or ""},
                        {"name": "TASK_CASE", "value": str(case_sensitive)},
                        {"name": "TASK_PIN", "value": pin},
                        {"name": "TASK_USER_ID", "value": user_id}
                    ]
                }]
            }
        )
        operation = client.run_job(request=request)
        print(f"🚀 Dispatched Cloud Job: {job_id}")
    except Exception as e:
        print(f"❌ Failed to dispatch Cloud job: {e}")
        # Mark as failed in DB so user isn't stuck forever
        db = firestore.Client(project=PROJECT_ID)
        db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': 'Cloud Dispatch Failed'})

# --- LOCAL CPU LOGIC ---
def check_key_batch(prefix, suffix, case_sensitive, batch_size):
    prefix_str = prefix or ""
    suffix_str = suffix or ""
    for i in range(batch_size):
        kp = Keypair()
        pub = str(kp.pubkey())
        p_match = pub.startswith(prefix_str) if case_sensitive else pub.lower().startswith(prefix_str.lower())
        s_match = pub.endswith(suffix_str) if case_sensitive else pub.lower().endswith(suffix_str.lower())
        if (not prefix or p_match) and (not suffix or s_match):
            return (str(kp.pubkey()), base58.b58encode(bytes(kp)).decode('ascii'), i + 1)
    return (None, None, batch_size)

def worker_batch_wrapper(args):
    return check_key_batch(*args)

def check_user_trials(user_id):
    try:
        db = firestore.Client(project=PROJECT_ID)
        docs = db.collection('vanity_jobs').where('user_id', '==', user_id).stream()
        return sum(1 for doc in docs if doc.to_dict().get('is_trial'))
    except: return 2

def verify_payment(signature, required_price_sol):
    if not signature: return False
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "getTransaction",
        "params": [signature, {"encoding": "json", "commitment": "confirmed", "maxSupportedTransactionVersion": 0}]
    }
    try:
        resp = requests.post(SOLANA_RPC_URL, json=payload, timeout=10)
        data = resp.json()
        if "error" in data or not data.get("result"): return False
        
        result = data.get("result")
        if result.get("meta", {}).get("err"): return False

        meta = result.get("meta", {})
        msg = result.get("transaction", {}).get("message", {})
        account_keys_raw = msg.get("accountKeys", [])
        accounts = [k.get("pubkey") if isinstance(k, dict) else k for k in account_keys_raw]

        try:
            idx = accounts.index(TREASURY_PUBKEY)
        except ValueError:
            print(f"Treasury not found in tx")
            return False

        received_sol = (meta.get("postBalances")[idx] - meta.get("preBalances")[idx]) / 1_000_000_000
        
        # Allow tiny variance for fees
        if received_sol < (required_price_sol - 0.0001):
            print(f"Insufficient: {received_sol} < {required_price_sol}")
            return False
        return True
    except Exception as e:
        print(f"Verify Error: {e}")
        return False

def get_deterministic_salt(user_id):
    return hashlib.sha256(user_id.encode()).digest()

def generate_key_from_pin(pin, user_id):
    salt = get_deterministic_salt(user_id)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))

def send_completion_email(to_email, job_id, public_key):
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = to_email
        msg['Subject'] = "VanityForge Job Complete!"
        body = f"<html><body><h2>Job Complete!</h2><p><strong>Public Key:</strong> {public_key}</p><p><a href='https://vanityforge.org'>Reveal Key</a></p></body></html>"
        msg.attach(MIMEText(body, 'html'))
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
        server.quit()
    except Exception as e: print(f"Email failed: {e}")

def background_grinder(job_id, user_id, prefix, suffix, case_sensitive, pin, email=None, notify=False):
    # DEDICATE TOTAL_GRINDING_CORES to this single worker
    pool_size = TOTAL_GRINDING_CORES
    try:
        db = firestore.Client(project=PROJECT_ID)
        checkpoint_ref = db.collection('job_checkpoints').document(job_id)
        doc = checkpoint_ref.get()
        total_attempts = doc.to_dict().get('last_attempt_number', 0) if doc.exists else 0
        last_save = total_attempts
        
        # NOTE: Status is already set to 'RUNNING' by the scheduler before calling this
        BATCH_SIZE = 50000

        with multiprocessing.Pool(processes=pool_size) as pool:
            pending = [pool.apply_async(worker_batch_wrapper, args=((prefix, suffix, case_sensitive, BATCH_SIZE),)) for _ in range(pool_size * 2)]
            found_key = None
            
            while not found_key:
                next_pending = []
                processed = 0
                for res in pending:
                    if res.ready():
                        processed += 1
                        try:
                            pub, raw, count = res.get()
                            total_attempts += count
                            if pub: 
                                found_key = (pub, raw)
                                break
                        except: pass
                    else:
                        next_pending.append(res)
                
                if found_key: break
                for _ in range(processed): next_pending.append(pool.apply_async(worker_batch_wrapper, args=((prefix, suffix, case_sensitive, BATCH_SIZE),)))
                pending = next_pending
                
                if total_attempts - last_save >= 10000000:
                    checkpoint_ref.set({'last_attempt_number': total_attempts})
                    last_save = total_attempts
                time.sleep(0.05)
            
            pool.terminate()
            pool.join()
            
            key = generate_key_from_pin(pin, user_id)
            enc_key = Fernet(key).encrypt(found_key[1].encode()).decode()
            
            db.collection('vanity_jobs').document(job_id).update({
                'status': 'COMPLETED', 'public_key': found_key[0], 'secret_key': enc_key, 'completed_at': firestore.SERVER_TIMESTAMP
            })
            if notify and email: send_completion_email(email, job_id, found_key[0])
            checkpoint_ref.delete()

    except Exception as e:
        try: firestore.Client(project=PROJECT_ID).collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': str(e)})
        except: pass

# --- QUEUE SCHEDULER (THE TRAFFIC CONTROLLER) ---
def scheduler_loop():
    """Background thread that checks for open slots and starts queued jobs."""
    print("✅ Queue Scheduler Started")
    while True:
        try:
            db = firestore.Client(project=PROJECT_ID)
            
            # 1. Count CURRENTLY running jobs
            running_cloud = 0
            running_local = 0
            running_docs = db.collection('vanity_jobs').where('status', '==', 'RUNNING').stream()
            for d in running_docs:
                if d.to_dict().get('is_cloud', False): running_cloud += 1
                else: running_local += 1
            
            # 2. Fetch QUEUED jobs (Oldest first = Fairness)
            # THIS QUERY REQUIRES THE COMPOSITE INDEX: status=QUEUED, order_by=created_at
            queued_docs = db.collection('vanity_jobs').where('status', '==', 'QUEUED').order_by('created_at').limit(10).stream()
            
            for doc in queued_docs:
                try:
                    data = doc.to_dict()
                    job_id = data.get('job_id')
                    is_cloud = data.get('is_cloud', False)

                    # Retrieve the temporary PIN needed for encryption
                    pin_plain = data.get('temp_pin')
                    if not pin_plain:
                        # Error state: Lost PIN? Fail job to clear queue
                        print(f"❌ Job {job_id} missing temp_pin, marking FAILED")
                        db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': 'PIN missing in queue'})
                        continue

                    # --- DISPATCH LOGIC ---
                    if is_cloud:
                        # RULE: Cloud jobs (Paid/Hard) -> Cloud Run Queue
                        if running_cloud < MAX_CLOUD_JOBS:
                            print(f"🚦 Dispatching Cloud Job: {job_id}")
                            db.collection('vanity_jobs').document(job_id).update({'status': 'RUNNING'})

                            # Cleanup temp PIN for security BEFORE dispatching
                            db.collection('vanity_jobs').document(job_id).update({'temp_pin': firestore.DELETE_FIELD})

                            # Trigger Cloud Run (AFTER security cleanup)
                            dispatch_cloud_job(job_id, data.get('user_id'), data.get('prefix'), data.get('suffix'), data.get('case_sensitive', True), pin_plain)

                            running_cloud += 1

                    else:
                        # RULE: Local jobs (Free/Easy) -> VM Queue
                        if running_local < MAX_LOCAL_JOBS:
                            print(f"🚦 Starting Local Job: {job_id}")
                            # Update status FIRST to reserve slot
                            db.collection('vanity_jobs').document(job_id).update({'status': 'RUNNING'})

                            # Start Local Process (NOTE: pin_plain is passed securely to the worker)
                            try:
                                p = multiprocessing.Process(target=background_grinder, args=(job_id, data.get('user_id'), data.get('prefix'), data.get('suffix'), data.get('case_sensitive', True), pin_plain, data.get('email'), data.get('notify')))
                                p.start()
                                running_local += 1
                            except Exception as pe:
                                print(f"❌ Failed to start process for {job_id}: {pe}")
                                db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': f"Process Start Failed: {pe}"})
                                continue

                            # Cleanup temp PIN
                            db.collection('vanity_jobs').document(job_id).update({'temp_pin': firestore.DELETE_FIELD})

                except Exception as e:
                    print(f"❌ Error processing job {doc.id}: {e}")
                    try:
                        db.collection('vanity_jobs').document(doc.id).update({'status': 'FAILED', 'error': str(e)})
                    except: pass

        except Exception as e:
            # The database index failure logs here.
            print(f"Scheduler Error: {e}")
        
        # Check queue every 5 seconds
        time.sleep(5)

@app.route('/check-user', methods=['POST'])
def check_user():
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    try:
        db = firestore.Client(project=PROJECT_ID)
        doc = db.collection('users').document(user_id).get()
        return jsonify({'has_pin': bool(doc.exists and doc.to_dict().get('pin_hash')), 'trials_used': check_user_trials(user_id)})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/set-pin', methods=['POST'])
def set_pin():
    data = request.get_json(silent=True) or {}
    user_id, pin = data.get('user_id'), data.get('pin')
    try:
        hashed = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
        firestore.Client(project=PROJECT_ID).collection('users').document(user_id).set({'user_id': user_id, 'pin_hash': hashed, 'updated_at': firestore.SERVER_TIMESTAMP}, merge=True)
        return jsonify({'message': 'PIN set'})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/submit-job', methods=['POST'])
def submit_job():
    data = request.get_json(silent=True) or {}
    user_id = data.get('userId') or data.get('user_id')
    prefix, suffix = data.get('prefix'), data.get('suffix', '')
    pin, tx_sig = data.get('pin'), data.get('transaction_signature')
    email, notify = data.get('email'), data.get('notify', False)
    
    if not user_id or not pin: return jsonify({'error': 'Missing ID or PIN'}), 400
    if not prefix and not suffix: return jsonify({'error': 'Prefix/Suffix required'}), 400
    if (prefix and not is_base58(prefix)) or (suffix and not is_base58(suffix)): return jsonify({'error': 'Invalid Base58'}), 400

    db = None
    try:
        # 1. Instantiate the client (always fresh)
        db = firestore.Client(project=PROJECT_ID)
        udoc = db.collection('users').document(user_id).get()
        if not udoc.exists or not udoc.to_dict().get('pin_hash'): return jsonify({'error': 'PIN not set'}), 400
        if not bcrypt.checkpw(pin.encode(), udoc.to_dict().get('pin_hash').encode()): return jsonify({'error': 'Invalid PIN'}), 401
        
        is_admin = (email == ADMIN_EMAIL)
        
        # --- DETERMINE JOB TYPE (CLOUD vs LOCAL) ---
        total_len = len(prefix or '') + len(suffix or '')
        
        # Prices Logic
        if total_len <= 4: base = 0.25
        elif total_len == 5: base = 0.50
        elif total_len == 6: base = 1.00
        elif total_len == 7: base = 2.00
        elif total_len == 8: base = 3.00
        else: base = 5.00
        
        # Free logic: <5 chars AND trials remaining
        price = 0.0 if (total_len <= 4 and check_user_trials(user_id) < 2) or is_admin else base * 0.5
        
        # Hard Limit
        if total_len > 8: return jsonify({'error': 'Max 8 chars allowed in Beta'}), 403

        # Payment Check
        if price > 0:
            if not tx_sig: return jsonify({'error': f'Payment of {price} SOL required'}), 402
            if not verify_payment(tx_sig, price): return jsonify({'error': 'Payment verification failed'}), 402

        job_id = str(uuid.uuid4())
        
        # --- JOB QUEUE LOGIC ---
        # 1. Determine "Hard" vs "Easy"
        is_cloud_job = (total_len >= 5) 

        # 2. Save Initial State as 'QUEUED'
        db.collection('vanity_jobs').document(job_id).set({
            'job_id': job_id, 
            'user_id': user_id, 
            'prefix': prefix, 
            'suffix': suffix, 
            'case_sensitive': data.get('case_sensitive', True),
            'status': 'QUEUED', 
            'is_cloud': is_cloud_job,
            'temp_pin': pin,
            'created_at': firestore.SERVER_TIMESTAMP, 
            'is_trial': (price == 0), 
            'price_sol': price,
            'transaction_signature': tx_sig, 
            'email': email, 
            'notify': notify
        })
        
        return jsonify({'job_id': job_id, 'message': 'Job Queued successfully'}), 202

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # 3. CRITICAL CLEANUP: Ensure the client connection is closed immediately after use
        cleanup_firestore_client(db)

@app.route('/reveal-key', methods=['POST'])
def reveal_key():
    data = request.get_json(silent=True) or {}
    job_id, pin = data.get('job_id'), data.get('pin')
    try:
        db = firestore.Client(project=PROJECT_ID)
        job = db.collection('vanity_jobs').document(job_id).get()
        if not job.exists: return jsonify({'error': 'Job not found'}), 404
        jdata = job.to_dict()
        
        # --- CRITICAL SECURITY FIX ---
        user_doc = db.collection('users').document(jdata['user_id']).get() 
        pin_hash = user_doc.to_dict().get('pin_hash', '') if user_doc.exists else ''

        # If the hash exists and the provided PIN does NOT match the hash, reject.
        if pin_hash and not bcrypt.checkpw(pin.encode(), pin_hash.encode()):
            return jsonify({'error': 'Invalid PIN'}), 401
        
        # PIN is correct for the job owner, proceed to decryption.
        key = generate_key_from_pin(pin, jdata['user_id'])
        return jsonify({'secret_key': Fernet(key).decrypt(jdata['secret_key'].encode()).decode()})
    except Exception as e: 
        print(f"Decryption Error: {e}")
        return jsonify({'error': 'Decryption failed'}), 400

if __name__ == '__main__':
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    # START THE TRAFFIC CONTROLLER THREAD
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()
    app.run(host='127.0.0.1', port=8080)
