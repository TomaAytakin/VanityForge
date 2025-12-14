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
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google.cloud import firestore
from google.cloud import run_v2
from solders.keypair import Keypair
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# 1. LOAD SECRETS
load_dotenv()

# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DYNAMIC CORE ALLOCATION ---
# Calculate cores for grinding, reserving 1 for the API/System
TOTAL_CORES = multiprocessing.cpu_count()
TOTAL_GRINDING_CORES = max(1, TOTAL_CORES - 1) 

# 2. CONFIGURATION
PROJECT_ID = os.getenv('PROJECT_ID', 'vanityforge').strip()
SOLANA_RPC_URL = os.getenv('SOLANA_RPC_URL', '').strip()
TREASURY_PUBKEY = os.getenv('TREASURY_PUBKEY', '').strip()
SMTP_EMAIL = os.getenv('SMTP_EMAIL', '').strip()
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '').strip()
os.environ.setdefault("SMTP_SERVER", "smtp.gmail.com")
os.environ.setdefault("SMTP_PORT", "587")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
ADMIN_EMAILS = {"tomaaytakin@gmail.com", "Jonny95hidalgo@gmail.com", "admin@vanityforge.org"}

# --- QUEUE LIMITS (TRAFFIC CONTROL) ---
MAX_LOCAL_JOBS = 1    # Max concurrent "Easy" jobs (Uses TOTAL_GRINDING_CORES)
MAX_CLOUD_JOBS = 50  # Max concurrent "Hard" jobs on Cloud Run

# CLOUD JOB CONFIGURATION
REDPANDA_JOB_NAME = os.getenv('REDPANDA_JOB_NAME', f"projects/{PROJECT_ID}/locations/us-central1/jobs/vanity-gpu-redpanda")
CPU_JOB_NAME = os.getenv('CPU_JOB_NAME', f"projects/{PROJECT_ID}/locations/europe-west1/jobs/vanity-gpu-worker")

# 3. SAFETY CHECK
if not SOLANA_RPC_URL or not TREASURY_PUBKEY or not SMTP_PASSWORD:
    logging.critical("CRITICAL ERROR: Missing environment variables. Make sure .env exists.")
    exit(1)

app = Flask(__name__, static_folder='.', static_url_path='', template_folder='.')
CORS(app, resources={r"/*": {"origins": ["https://tomaaytakin.github.io", "https://vanityforge.org"]}})

# --- RATE LIMITER CONFIGURATION ---
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://"
)

@app.route('/')
def index(): return app.send_static_file('index.html')

@app.route('/roadmap')
def roadmap(): return app.send_static_file('roadmap.html')

@app.route('/vvvip')
def vvvip():
    rpc_url = os.getenv("SOLANA_RPC_URL", "https://rpc.ankr.com/solana")
    return render_template('vvvip.html', rpc_url=rpc_url)

@app.route('/faq')
def faq(): return app.send_static_file('faq.html')

def cleanup_firestore_client(db):
    """Safely closes the Firestore client connection."""
    try:
        if db:
            db.close()
    except Exception:
        # Fail silently during cleanup to keep logs clean
        pass

def is_base58(s):
    if not s: return True
    return bool(re.match(r'^[1-9A-HJ-NP-Za-km-z]+$', s))

# --- DISPATCHER LOGIC (CLOUD RUN JOBS) ---
def dispatch_cloud_job(job_id, user_id, prefix, suffix, case_sensitive, pin, worker_type):
    """Triggers the remote Cloud Run Job (CPU or GPU)."""
    try:
        # Determine Target Job
        if worker_type == "cloud-run-gpu-redpanda":
            target_job = REDPANDA_JOB_NAME
            logging.info("🏎️ Dispatching to RedPanda GPU Lane")
        else:
            target_job = CPU_JOB_NAME
            logging.info("🚙 Dispatching to Standard CPU Lane")

        client = run_v2.JobsClient()
        request = run_v2.RunJobRequest(
            name=target_job,
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
        logging.info(f"🚀 Dispatched Cloud Job: {job_id}")
    except Exception as e:
        logging.exception(f"❌ Failed to dispatch Cloud job")
        # Mark as failed in DB so user isn't stuck forever
        db = firestore.Client(project=PROJECT_ID)
        try:
            db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': 'Cloud Dispatch Failed'})
        finally:
            cleanup_firestore_client(db)

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
    db = None
    try:
        db = firestore.Client(project=PROJECT_ID)
        docs = db.collection('vanity_jobs').where('user_id', '==', user_id).stream()
        return sum(1 for doc in docs if doc.to_dict().get('is_trial'))
    except: return 2
    finally:
        cleanup_firestore_client(db)

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
            logging.warning(f"Treasury not found in tx")
            return False

        received_sol = (meta.get("postBalances")[idx] - meta.get("preBalances")[idx]) / 1_000_000_000
        
        # Allow tiny variance for fees
        if received_sol < (required_price_sol - 0.0001):
            logging.warning(f"Insufficient: {received_sol} < {required_price_sol}")
            return False
        return True
    except Exception as e:
        logging.exception(f"Verify Error")
        return False

def get_deterministic_salt(user_id):
    return hashlib.sha256(user_id.encode()).digest()

def generate_key_from_pin(pin, user_id):
    salt = get_deterministic_salt(user_id)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))

# --- SOLA EMAIL ENGINE ---
SOLA_BANNER = "https://raw.githubusercontent.com/TomaAytakin/VanityForge/main/assets/Email%20Assets/vfemailbanner.png"
SOLA_PAW = "https://raw.githubusercontent.com/TomaAytakin/VanityForge/main/assets/Email%20Assets/vfemailsig.png"

def get_sola_html(body):
    """Wraps content in the Sola/Red Panda Branding"""
    return f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; color: #333;">
        <img src="{SOLA_BANNER}" width="100%" style="border-radius: 8px 8px 0 0;" alt="VanityForge Banner">
        <div style="padding: 20px; background-color: #ffffff; border: 1px solid #e0e0e0; border-top: none;">
            {body}
        </div>
        <div style="padding: 20px; background-color: #f9f9f9; border: 1px solid #e0e0e0; border-top: none; border-radius: 0 0 8px 8px;">
            <table style="width: 100%;">
                <tr>
                    <td style="width: 60px; vertical-align: middle;">
                        <img src="{SOLA_PAW}" width="50" height="50" alt="Paw">
                    </td>
                    <td style="vertical-align: middle;">
                        <strong style="color: #E55039; font-size: 16px;">Sola</strong> 🐼<br>
                        <span style="color: #666; font-size: 12px;">Chief Forging Scout @ VanityForge</span><br>
                        <a href="https://vanityforge.org" style="color: #6c5ce7; text-decoration: none; font-size: 12px;">vanityforge.org</a>
                    </td>
                </tr>
            </table>
        </div>
    </div>
    """

def send_email_wrapper(to_email, subject, html_content):
    """Handles the Alias Spoofing (Login as Admin, Send as Support)"""
    try:
        # Explicitly set the 'From' header to vanity address while logging in with SMTP creds
        sender_display = "VanityForge Support <support@vanityforge.org>"
        login_email = os.getenv('SMTP_EMAIL')
        login_password = os.getenv('SMTP_PASSWORD')

        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_display
        msg['To'] = to_email

        msg.attach(MIMEText(html_content, 'html'))

        context = ssl.create_default_context()
        with smtplib.SMTP(os.getenv('SMTP_SERVER'), int(os.getenv('SMTP_PORT'))) as server:
            server.starttls(context=context)
            server.login(login_email, login_password)
            # Send using login_email as envelope sender, but msg['From'] is displayed to user
            server.sendmail(login_email, to_email, msg.as_string())

        logging.info(f"📧 Sola Email sent to {to_email}")
        return True
    except Exception as e:
        logging.error(f"❌ Email Failed: {e}")
        return False

def send_start_email(to_email, job_id, prefix, suffix, price, is_trial):
    # Calculate Pricing Logic for Receipt
    total_len = len(prefix or '') + len(suffix or '')
    if total_len <= 4: base = 0.25
    elif total_len == 5: base = 0.50
    elif total_len == 6: base = 1.00
    elif total_len == 7: base = 2.00
    elif total_len == 8: base = 3.00
    else: base = 5.00

    # Standard Beta Price is 50% of Base
    standard_price = base * 0.5

    if is_trial:
        # Free Trial: Show strikethrough original price -> 0 SOL
        price_display = f"<s>{standard_price} SOL</s> 0 SOL (Free Trial)"
    else:
        # Paid Job: Show actual paid amount
        price_display = f"{price} SOL"

    receipt_html = f"""
    <h2 style="color: #2d3436;">Forge Started! ⚒️</h2>
    <p>Hey Degen! 🐾</p>
    <p>Sola here. I've successfully dispatched your job to the grinder.</p>
    <div style="background: #eee; padding: 10px; border-radius: 5px; margin: 15px 0;">
        <strong>Target:</strong> ...{suffix} (or {prefix}...)<br>
        <strong>Job ID:</strong> {job_id}<br>
        <strong>Price:</strong> {price_display}
    </div>
    <p>Sit tight! I'll ping you the second it's ready.</p>
    """
    send_email_wrapper(to_email, "Forge Started: Your Custom Wallet is Brewing! ⚒️", get_sola_html(receipt_html))

def send_completion_email(to_email, public_key, private_key_enc=None):
    success_html = f"""
    <h2 style="color: #00b894;">Forge Complete! 🚀</h2>
    <p>Great news from the bamboo forest! 🎋</p>
    <p>Your custom vanity wallet has been found.</p>
    <div style="border: 2px dashed #00b894; padding: 15px; text-align: center; margin: 20px 0;">
        <strong>Public Key:</strong><br>
        <code style="font-size: 14px; word-break: break-all;">{public_key}</code>
    </div>
    <p style="text-align: center;">
        <a href="https://vanityforge.org" style="background-color: #6c5ce7; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; font-weight: bold;">Reveal Private Key</a>
    </p>
    """
    send_email_wrapper(to_email, "Forge Complete! Your Wallet is Ready 🚀", get_sola_html(success_html))

def background_grinder(job_id, user_id, prefix, suffix, case_sensitive, pin, email=None, notify=False):
    # DEDICATE TOTAL_GRINDING_CORES to this single worker
    pool_size = TOTAL_GRINDING_CORES
    db = None
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
            if notify and email: send_completion_email(email, found_key[0])
            checkpoint_ref.delete()

    except Exception as e:
        logging.exception("Background grinder failed")
        if db:
            try: db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': str(e)})
            except: pass
    finally:
        cleanup_firestore_client(db)

# --- QUEUE SCHEDULER (THE TRAFFIC CONTROLLER) ---
def scheduler_loop():
    """Background thread that checks for open slots and starts queued jobs."""
    logging.info("✅ Queue Scheduler Started")
    db = firestore.Client(project=PROJECT_ID)
    while True:
        try:
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
                        logging.error(f"❌ Job {job_id} missing temp_pin, marking FAILED")
                        db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': 'PIN missing in queue'})
                        continue

                    # --- DISPATCH LOGIC ---
                    if is_cloud:
                        # RULE: Cloud jobs (Paid/Hard) -> Cloud Run Queue
                        if running_cloud < MAX_CLOUD_JOBS:
                            logging.info(f"🚦 Dispatching Cloud Job: {job_id}")
                            db.collection('vanity_jobs').document(job_id).update({'status': 'RUNNING'})

                            if data.get('notify') and data.get('email'):
                                try:
                                    send_start_email(data['email'], job_id, data.get('prefix'), data.get('suffix'), data.get('price_sol'), data.get('is_trial'))
                                except Exception as e:
                                    logging.error(f"Failed to send start email: {e}")

                            # Cleanup temp PIN for security BEFORE dispatching
                            db.collection('vanity_jobs').document(job_id).update({'temp_pin': firestore.DELETE_FIELD})

                            # Trigger Cloud Run (AFTER security cleanup)
                            dispatch_cloud_job(
                                job_id, 
                                data.get('user_id'), 
                                data.get('prefix'), 
                                data.get('suffix'), 
                                data.get('case_sensitive', True), 
                                pin_plain,
                                data.get('worker_type', 'cloud-run-cpu') # Default to CPU if missing
                            )

                            running_cloud += 1

                    else:
                        # RULE: Local jobs (Free/Easy) -> VM Queue
                        if running_local < MAX_LOCAL_JOBS:
                            logging.info(f"🚦 Starting Local Job: {job_id}")
                            # Update status FIRST to reserve slot
                            db.collection('vanity_jobs').document(job_id).update({'status': 'RUNNING'})

                            if data.get('notify') and data.get('email'):
                                try:
                                    send_start_email(data['email'], job_id, data.get('prefix'), data.get('suffix'), data.get('price_sol'), data.get('is_trial'))
                                except Exception as e:
                                    logging.error(f"Failed to send start email: {e}")

                            # Start Local Process (NOTE: pin_plain is passed securely to the worker)
                            try:
                                p = multiprocessing.Process(target=background_grinder, args=(job_id, data.get('user_id'), data.get('prefix'), data.get('suffix'), data.get('case_sensitive', True), pin_plain, data.get('email'), data.get('notify')))
                                p.start()
                                running_local += 1
                            except Exception as pe:
                                logging.exception(f"❌ Failed to start process for {job_id}")
                                db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': f"Process Start Failed: {pe}"})
                                continue

                            # Cleanup temp PIN
                            db.collection('vanity_jobs').document(job_id).update({'temp_pin': firestore.DELETE_FIELD})

                except Exception as e:
                    logging.exception(f"❌ Error processing job {doc.id}")
                    try:
                        db.collection('vanity_jobs').document(doc.id).update({'status': 'FAILED', 'error': str(e)})
                    except: pass

        except Exception as e:
            # The database index failure logs here.
            logging.exception(f"Scheduler Error")
        
        # Check queue every 5 seconds
        time.sleep(5)

@app.route('/check-user', methods=['POST'])
def check_user():
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    db = None
    try:
        db = firestore.Client(project=PROJECT_ID)
        doc = db.collection('users').document(user_id).get()
        return jsonify({'has_pin': bool(doc.exists and doc.to_dict().get('pin_hash')), 'trials_used': check_user_trials(user_id)})
    except Exception as e: return jsonify({'error': str(e)}), 500
    finally:
        cleanup_firestore_client(db)

@app.route('/set-pin', methods=['POST'])
def set_pin():
    data = request.get_json(silent=True) or {}
    user_id, pin = data.get('user_id'), data.get('pin')
    db = None
    try:
        hashed = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
        db = firestore.Client(project=PROJECT_ID)
        db.collection('users').document(user_id).set({'user_id': user_id, 'pin_hash': hashed, 'updated_at': firestore.SERVER_TIMESTAMP}, merge=True)
        return jsonify({'message': 'PIN set'})
    except Exception as e: return jsonify({'error': str(e)}), 500
    finally:
        cleanup_firestore_client(db)

@app.route('/submit-job', methods=['POST'])
@limiter.limit("60 per minute")
def submit_job():
    data = request.get_json(silent=True) or {}
    user_id = data.get('userId') or data.get('user_id')
    prefix, suffix = data.get('prefix'), data.get('suffix', '')
    pin, tx_sig = data.get('pin'), data.get('transaction_signature')
    email, notify = data.get('email'), data.get('notify', False)
    use_gpu = data.get('use_gpu', False)
    
    if not user_id or not pin: return jsonify({'error': 'Missing ID or PIN'}), 400
    if not prefix and not suffix: return jsonify({'error': 'Prefix/Suffix required'}), 400
    if (prefix and not is_base58(prefix)) or (suffix and not is_base58(suffix)): return jsonify({'error': 'Invalid Base58'}), 400

    db = None
    try:
        # 1. Instantiate the client (always fresh)
        db = firestore.Client(project=PROJECT_ID)

        udoc = db.collection('users').document(user_id).get()
        if not udoc.exists or not udoc.to_dict().get('pin_hash'): return jsonify({'error': 'PIN not set'}), 400

        user_data = udoc.to_dict()
        
        # --- START ADMIN BYPASS FIX ---
        is_admin_email = email in ADMIN_EMAILS
        is_god_mode = user_data.get('god_mode', False) or is_admin_email
        # --- END ADMIN BYPASS FIX ---

        # --- ACTIVE JOB CHECK ---
        if not is_god_mode: # ADMINS NOW BYPASS THIS CHECK
            active_jobs = db.collection('vanity_jobs') \
                .where('user_id', '==', user_id) \
                .where('status', 'in', ['QUEUED', 'RUNNING']) \
                .limit(1).stream()

            if any(active_jobs):
                return jsonify({'error': 'You have an active job'}), 400

        if not bcrypt.checkpw(pin.encode(), user_data.get('pin_hash').encode()): return jsonify({'error': 'Invalid PIN'}), 401
        
        is_admin = (email in ADMIN_EMAILS)
        
        # --- DETERMINE JOB TYPE (CLOUD vs LOCAL) ---
        total_len = len(prefix or '') + len(suffix or '')
        
        # Automatic GPU Routing > 5
        if total_len >= 6:
            use_gpu = True

        # Prices Logic
        if total_len <= 4: base = 0.25
        elif total_len == 5: base = 0.50
        elif total_len == 6: base = 1.00
        elif total_len == 7: base = 2.00
        elif total_len == 8: base = 3.00
        else: base = 5.00

        # GPU Surcharge
        if use_gpu:
            base = base * 1.5
        
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
        # Determine Worker Type & Cloud Status
        if use_gpu:
            worker_type = "cloud-run-gpu-redpanda"
            is_cloud_job = True
        elif total_len >= 5:
            worker_type = "cloud-run-cpu"
            is_cloud_job = True
        else:
            worker_type = "local-cpu"
            is_cloud_job = False

        # 2. Save Initial State as 'QUEUED'
        db.collection('vanity_jobs').document(job_id).set({
            'job_id': job_id, 
            'user_id': user_id, 
            'prefix': prefix, 
            'suffix': suffix, 
            'case_sensitive': data.get('case_sensitive', True),
            'status': 'QUEUED', 
            'is_cloud': is_cloud_job,
            'worker_type': worker_type,
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
        logging.exception("Submit job failed")
        return jsonify({'error': str(e)}), 500

    finally:
        # 3. CRITICAL CLEANUP: Ensure the client connection is closed immediately after use
        cleanup_firestore_client(db)

@app.route('/reveal-key', methods=['POST'])
def reveal_key():
    data = request.get_json(silent=True) or {}
    job_id, pin = data.get('job_id'), data.get('pin')
    db = None
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
        logging.exception("Decryption Error")
        return jsonify({'error': 'Decryption failed'}), 400
    finally:
        cleanup_firestore_client(db)

@app.route('/api/check-sns', methods=['POST'])
def check_sns():
    """Proxy route to check SNS availability via Mainnet RPC"""
    data = request.get_json(silent=True) or {}
    public_key = data.get('publicKey')

    if not public_key:
        return jsonify({'error': 'Missing publicKey'}), 400

    # Payload for getAccountInfo
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [
            public_key,
            {"encoding": "base58"}
        ]
    }

    try:
        # Use the standard mainnet-beta endpoint as requested
        # Server-side request avoids browser CORS issues
        rpc_url = "https://api.mainnet-beta.solana.com"
        resp = requests.post(rpc_url, json=payload, timeout=5)

        # If the RPC returns an error (even 200 OK with error body), forward it or handle it
        if resp.status_code != 200:
            logging.error(f"SNS Proxy RPC Error: {resp.status_code} {resp.text}")
            return jsonify({'error': 'RPC Error'}), resp.status_code

        rpc_data = resp.json()
        return jsonify(rpc_data)

    except Exception as e:
        logging.exception("SNS Proxy Failed")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    # START THE TRAFFIC CONTROLLER THREAD
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()
    app.run(host='127.0.0.1', port=8080)
