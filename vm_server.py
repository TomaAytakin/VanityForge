import os
import uuid
import logging
import multiprocessing
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
from solders.keypair import Keypair
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configuration
PROJECT_ID = 'vanityforge'
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
TREASURY_PUBKEY = "2d8W2qgRi7BCMTuyv2ZHbf9zJi5C1hT4VRudRnxZifVD"
ADMIN_EMAIL = "tomaaytakin@gmail.com"

# Email Configuration Placeholders
SMTP_SERVER = "smtp.gmail.com" # Placeholder
SMTP_PORT = 587
SMTP_EMAIL = "your_email@gmail.com" # Placeholder
SMTP_PASSWORD = "your_password" # Placeholder

# Serve static files from the current directory
app = Flask(__name__, static_folder='.', static_url_path='')
# Strict CORS: Only allow requests from the specific frontend URL
CORS(app, resources={r"/*": {"origins": "https://tomaaytakin.github.io"}})

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/roadmap')
def roadmap():
    return app.send_static_file('roadmap.html')

@app.route('/faq')
def faq():
    return app.send_static_file('faq.html')

# Global to track active jobs
MAX_CONCURRENT_JOBS = 6
# Reserve 2 core slots for "Quick Jobs" (estimated time < 15 minutes).
# Logic:
# Long jobs can only acquire from sem_general (size 4).
# Quick jobs try sem_general first, if full, they use sem_reserved (size 2).
sem_general = multiprocessing.Semaphore(4)
sem_reserved = multiprocessing.Semaphore(2)

def is_base58(s):
    """Checks if the string contains only Base58 characters."""
    if not s: return True
    # Base58 alphabet: 1-9, A-H, J-N, P-Z, a-k, m-z
    pattern = r'^[1-9A-HJ-NP-Za-km-z]+$'
    return bool(re.match(pattern, s))

def check_key_batch(prefix, suffix, case_sensitive, batch_size):
    """
    Checks a batch of keys.
    Returns (public_key, secret_key, attempts_made)
    """
    for i in range(batch_size):
        kp = Keypair()
        pub = str(kp.pubkey())

        # Prefix Check
        if case_sensitive:
            p_match = pub.startswith(prefix) if prefix else True
        else:
            p_match = pub.lower().startswith(prefix.lower()) if prefix else True

        if not p_match:
            continue

        # Suffix Check
        if case_sensitive:
            s_match = pub.endswith(suffix) if suffix else True
        else:
            s_match = pub.lower().endswith(suffix.lower()) if suffix else True

        if s_match:
            # Found match
            return (str(kp.pubkey()), base58.b58encode(bytes(kp)).decode('ascii'), i + 1)

    return (None, None, batch_size)

def worker_batch_wrapper(args):
    """
    Wrapper to unpack arguments and run check_key_batch.
    """
    prefix, suffix, case_sensitive, batch_size = args
    return check_key_batch(prefix, suffix, case_sensitive, batch_size)

def check_user_trials(user_id):
    """
    Checks how many trial jobs the user has used.
    Returns the count of used trials.
    """
    try:
        db = firestore.Client(project=PROJECT_ID)
        # Retrieve all jobs for the user and count those marked as trials
        # We fetch all jobs for the user to ensure accuracy without complex composite indexes
        docs = db.collection('vanity_jobs').where('user_id', '==', user_id).stream()

        trial_count = 0
        for doc in docs:
            data = doc.to_dict()
            if data.get('is_trial'):
                trial_count += 1
        return trial_count
    except Exception as e:
        print(f"Error checking trials: {e}")
        return 2 # Fail safe: assume no trials left if error

def verify_payment(signature):
    """Verifies that a transaction is successful on-chain."""
    if not signature: return False

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [
            signature,
            {"encoding": "json", "commitment": "confirmed", "maxSupportedTransactionVersion": 0}
        ]
    }

    try:
        resp = requests.post(SOLANA_RPC_URL, json=payload, timeout=10)
        data = resp.json()

        if "error" in data:
            print(f"RPC Error for {signature}: {data['error']}")
            return False

        result = data.get("result")
        if not result:
            return False

        if result.get("meta", {}).get("err"):
             return False # Transaction failed

        # In a full implementation, we would parse 'transaction' -> 'message' -> 'accountKeys'
        # and 'meta' -> 'postBalances' - 'preBalances' to verify amount transferred to TREASURY_PUBKEY.

        return True
    except Exception as e:
        print(f"Verification Exception: {e}")
        return False

def get_deterministic_salt(user_id):
    """Generates a deterministic salt based on the user_id."""
    return hashlib.sha256(user_id.encode()).digest()

def generate_key_from_pin(pin, user_id):
    """
    Generates a Fernet key from the user's PIN using a deterministic salt based on user_id.
    """
    salt = get_deterministic_salt(user_id)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))

def send_completion_email(to_email, job_id, public_key):
    """Sends an email notification upon job completion."""
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = to_email
        msg['Subject'] = "VanityForge Job Complete!"

        body = f"""
        <html>
          <body>
            <h2>Your VanityForge Job is Complete!</h2>
            <p><strong>Public Key:</strong> {public_key}</p>
            <p>Your private key has been securely generated and encrypted.</p>
            <p><a href="https://tomaaytakin.github.io">Log in to VanityForge</a> to reveal your private key.</p>
            <br>
            <p><em>Job ID: {job_id}</em></p>
          </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
        server.quit()
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def background_grinder(job_id, user_id, prefix, suffix, case_sensitive, pin, is_quick, email=None, notify=False):
    """
    Background process that manages the Pool of workers.
    """
    # CPU Affinity: Limit to 3 processes to keep one core free
    pool_size = 3
    acquired_general = False
    acquired_reserved = False
    try:
        # Concurrency Logic: Wait for a slot
        if is_quick:
            # Try general slot first (non-blocking)
            if sem_general.acquire(block=False):
                acquired_general = True
            else:
                # If general full, block on reserved slot
                sem_reserved.acquire() # Blocking
                acquired_reserved = True
        else:
            # Long jobs must wait for general slot
            sem_general.acquire() # Blocking
            acquired_general = True

        # Initialize Firestore inside the process
        db = firestore.Client(project=PROJECT_ID)

        # Checkpointing Initialization
        checkpoint_ref = db.collection('job_checkpoints').document(job_id)
        doc = checkpoint_ref.get()
        total_attempts = 0
        if doc.exists:
            total_attempts = doc.to_dict().get('last_attempt_number', 0)
            # Resuming logic: We just continue counting from here.

        last_checkpoint_save = total_attempts

        # Update status to RUNNING
        db.collection('vanity_jobs').document(job_id).update({'status': 'RUNNING'})

        # Initialize Pool with limit
        BATCH_SIZE = 50000 # Process keys in chunks

        with multiprocessing.Pool(processes=pool_size) as pool:
            pending_results = []

            # Helper to add tasks
            def add_task():
                return pool.apply_async(worker_batch_wrapper, args=((prefix, suffix, case_sensitive, BATCH_SIZE),))

            # Initial fill
            for _ in range(pool_size * 2):
                pending_results.append(add_task())

            found_key = None

            while not found_key:
                next_pending = []
                results_processed = 0

                for res in pending_results:
                    if res.ready():
                        results_processed += 1
                        try:
                            pub, raw_sec, count = res.get()
                            total_attempts += count

                            if pub:
                                found_key = (pub, raw_sec)
                                break
                        except Exception as e:
                            print(f"Worker error: {e}")
                    else:
                        next_pending.append(res)

                if found_key:
                    break

                # Refill pool to keep it saturated
                for _ in range(results_processed):
                    next_pending.append(add_task())

                pending_results = next_pending

                # Progress Save: Every 10 million attempts
                if total_attempts - last_checkpoint_save >= 10000000:
                    checkpoint_ref.set({'last_attempt_number': total_attempts})
                    last_checkpoint_save = total_attempts

                time.sleep(0.05) # Prevent busy loop

            # Cleanup
            pool.terminate()
            pool.join()

            public_key, raw_secret_key = found_key

            # Encrypt the private key
            # Use deterministic salt based on user_id + PIN
            key = generate_key_from_pin(pin, user_id)
            f = Fernet(key)
            encrypted_secret_key = f.encrypt(raw_secret_key.encode()).decode()

            # Save to Firestore
            db.collection('vanity_jobs').document(job_id).update({
                'status': 'COMPLETED',
                'public_key': public_key,
                'secret_key': encrypted_secret_key,
                'completed_at': firestore.SERVER_TIMESTAMP
            })
            print(f"SECRET KEY SAVED for {job_id}")

            # Send Email Notification
            if notify and email:
                send_completion_email(email, job_id, public_key)

            # Final Step: Delete checkpoint
            checkpoint_ref.delete()

    except Exception as e:
        print(f"Error in background grinder: {e}")
        try:
             db = firestore.Client(project=PROJECT_ID)
             db.collection('vanity_jobs').document(job_id).update({
                'status': 'FAILED',
                'error': str(e)
            })
        except:
            pass
    finally:
        # Release semaphore
        if acquired_general:
            sem_general.release()
        if acquired_reserved:
            sem_reserved.release()

@app.route('/check-user', methods=['POST'])
def check_user():
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400

    try:
        db = firestore.Client(project=PROJECT_ID)
        doc = db.collection('users').document(user_id).get()

        trials_used = check_user_trials(user_id)

        has_pin = False
        if doc.exists and doc.to_dict().get('pin_hash'):
            has_pin = True

        return jsonify({'has_pin': has_pin, 'trials_used': trials_used})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/set-pin', methods=['POST'])
def set_pin():
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    pin = data.get('pin')

    if not user_id or not pin:
        return jsonify({'error': 'Missing user_id or pin'}), 400

    # Hash the PIN
    # bcrypt automatically handles salt
    hashed = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()

    try:
        db = firestore.Client(project=PROJECT_ID)
        db.collection('users').document(user_id).set({
            'user_id': user_id,
            'pin_hash': hashed,
            'updated_at': firestore.SERVER_TIMESTAMP
        }, merge=True)
        return jsonify({'message': 'PIN set successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/submit-job', methods=['POST'])
def submit_job():
    data = request.get_json(silent=True) or {}
    user_id = data.get('userId') or data.get('user_id')
    prefix = data.get('prefix')
    suffix = data.get('suffix', '')
    case_sensitive = data.get('case_sensitive', True)
    pin = data.get('pin')
    transaction_signature = data.get('transaction_signature')
    email = data.get('email')
    notify = data.get('notify', False)

    is_admin = (email == ADMIN_EMAIL)

    # Input Validation
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
    if not prefix and not suffix:
        return jsonify({'error': 'Prefix or suffix required'}), 400
    if not pin:
        return jsonify({'error': 'PIN is mandatory'}), 400

    if (prefix and not is_base58(prefix)) or (suffix and not is_base58(suffix)):
        return jsonify({'error': 'Invalid characters. Base58 characters only.'}), 400

    # Verify PIN first
    try:
        db = firestore.Client(project=PROJECT_ID)
        user_doc = db.collection('users').document(user_id).get()
        if not user_doc.exists:
             return jsonify({'error': 'User not found. Please refresh.'}), 400

        stored_hash = user_doc.to_dict().get('pin_hash')
        if not stored_hash:
             return jsonify({'error': 'PIN not set. Please set a PIN.'}), 400

        if not bcrypt.checkpw(pin.encode(), stored_hash.encode()):
             return jsonify({'error': 'Invalid PIN.'}), 401

        # Check User Limit: Return 403 if user has QUEUED or RUNNING job
        if not is_admin:
            active_jobs = db.collection('vanity_jobs').where('user_id', '==', user_id).where('status', 'in', ['QUEUED', 'RUNNING']).get()
            if len(list(active_jobs)) > 0:
                return jsonify({'error': 'You can only run one job at a time.'}), 403

    except Exception as e:
         return jsonify({'error': str(e)}), 500

    # Calculate Price and Check Trials
    total_len = len(prefix or '') + len(suffix or '')

    # Hard Limit Enforcement (Beta) - Bypassed for Admin
    if not is_admin and total_len > 5:
        return jsonify({'error': 'Maximum 5 characters allowed in Beta.'}), 403

    # Fixed Pricing Logic (Pre-Discount)
    if total_len <= 4:
        base_price = 0.25
    elif total_len == 5:
        base_price = 0.50
    elif total_len == 6:
        base_price = 1.00
    elif total_len == 7:
        base_price = 2.00
    elif total_len == 8:
        base_price = 3.00
    else:
        base_price = 5.00

    # Apply 50% Beta Discount
    price_sol = base_price * 0.5

    # Check Trials
    trials_used = check_user_trials(user_id)
    is_trial = False

    if total_len <= 4 and trials_used < 2:
        is_trial = True
        price_sol = 0.0

    if is_admin:
        price_sol = 0.0
        is_trial = True # Treat as free

    # Trial Enforcement: Reject if cost is 0 but trials used up (Safety check) - Bypassed for Admin
    if not is_admin and price_sol == 0 and trials_used >= 2:
         return jsonify({'error': 'Trial limit reached. Please pay to continue.'}), 403

    # Payment Verification - Bypassed for Admin
    if not is_admin and price_sol > 0:
        if not transaction_signature:
             return jsonify({'error': f'Payment of {price_sol:.4f} SOL required.'}), 402

        if not verify_payment(transaction_signature):
             return jsonify({'error': 'Payment verification failed. Transaction invalid or not confirmed.'}), 402

    # Time (seconds) = (0.5 * 58^TotalLength) / 5,000,000
    est_seconds = (0.5 * (58 ** total_len)) / 5000000
    is_quick = est_seconds < 900 # 15 minutes

    job_id = str(uuid.uuid4())

    # Initial DB Record
    try:
        db = firestore.Client(project=PROJECT_ID)
        db.collection('vanity_jobs').document(job_id).set({
            'job_id': job_id,
            'user_id': user_id,
            'prefix': prefix,
            'suffix': suffix,
            'case_sensitive': case_sensitive,
            'status': 'QUEUED',
            'created_at': firestore.SERVER_TIMESTAMP,
            'is_trial': is_trial,
            'price_sol': price_sol,
            'transaction_signature': transaction_signature,
            'email': email,
            'notify': notify
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Spawn Background Process
    p = multiprocessing.Process(target=background_grinder, args=(job_id, user_id, prefix, suffix, case_sensitive, pin, is_quick, email, notify))
    p.start()

    return jsonify({'job_id': job_id, 'message': 'Job accepted', 'is_trial': is_trial, 'price_sol': price_sol}), 202

@app.route('/reveal-key', methods=['POST'])
def reveal_key():
    data = request.get_json(silent=True) or {}
    job_id = data.get('job_id')
    pin = data.get('pin')

    if not job_id or not pin:
        return jsonify({'error': 'Missing job_id or pin'}), 400

    try:
        db = firestore.Client(project=PROJECT_ID)
        job_doc = db.collection('vanity_jobs').document(job_id).get()
        if not job_doc.exists:
             return jsonify({'error': 'Job not found'}), 404

        job_data = job_doc.to_dict()
        user_id = job_data.get('user_id')
        encrypted_secret_key = job_data.get('secret_key')

        if not encrypted_secret_key:
            return jsonify({'error': 'Job not completed or key not found'}), 400

        # Verify PIN against User DB to ensure it's the correct user
        user_doc = db.collection('users').document(user_id).get()
        if user_doc.exists:
             stored_hash = user_doc.to_dict().get('pin_hash')
             if stored_hash and not bcrypt.checkpw(pin.encode(), stored_hash.encode()):
                 return jsonify({'error': 'Invalid PIN'}), 401

        # Derive Key and Decrypt
        key = generate_key_from_pin(pin, user_id)
        f = Fernet(key)

        try:
            raw_secret_key = f.decrypt(encrypted_secret_key.encode()).decode()
            return jsonify({'secret_key': raw_secret_key})
        except Exception:
            return jsonify({'error': 'Decryption failed. Did you change your PIN?'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Reap zombie processes automatically
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    # Bind to 127.0.0.1 (localhost) on port 8080 as requested for Caddy reverse proxy
    # This ensures ONLY Caddy can talk to the app, blocking direct external access.
    app.run(host='127.0.0.1', port=8080)
