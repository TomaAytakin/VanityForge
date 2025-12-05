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
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import firestore
from solders.keypair import Keypair
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configuration
PROJECT_ID = 'vanityforge'

# Serve static files from the current directory
app = Flask(__name__, static_folder='.', static_url_path='')
# Strict CORS: Only allow requests from the specific frontend URL
CORS(app, resources={r"/*": {"origins": "https://tomaaytakin.github.io"}})

@app.route('/')
def index():
    return app.send_static_file('index.html')

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

def check_key(prefix, suffix, case_sensitive):
    """
    Worker function to run in a Pool.
    Tries a batch of keys.
    """
    # Check a batch of keys to reduce overhead of inter-process communication
    BATCH_SIZE = 500
    for _ in range(BATCH_SIZE):
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
            return (str(kp.pubkey()), base58.b58encode(bytes(kp)).decode('ascii'))

    return None

def worker_wrapper(args):
    """
    Wrapper to unpack arguments and run check_key loop.
    """
    prefix, suffix, case_sensitive = args
    # Loop indefinitely until found (or terminated by parent)
    while True:
        result = check_key(prefix, suffix, case_sensitive)
        if result:
            return result

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

def background_grinder(job_id, user_id, prefix, suffix, case_sensitive, pin, is_quick):
    """
    Background process that manages the Pool of workers.
    """
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

        # Update status to RUNNING
        db.collection('vanity_jobs').document(job_id).update({'status': 'RUNNING'})

        # Initialize Pool with exactly 4 processes
        pool_size = 4
        with multiprocessing.Pool(processes=pool_size) as pool:
            results = []
            # Submit tasks to the pool
            for _ in range(pool_size):
                results.append(pool.apply_async(worker_wrapper, args=((prefix, suffix, case_sensitive),)))

            # Monitor results
            found_key = None
            while not found_key:
                for res in results:
                    if res.ready():
                        found_key = res.get()
                        if found_key:
                            break
                if found_key:
                    break
                time.sleep(0.1) # Prevent busy loop

            # Found key, terminate pool
            pool.terminate()
            pool.join()

            public_key, raw_secret_key = found_key

            # Encrypt the private key
            # Use deterministic salt based on user_id + PIN
            key = generate_key_from_pin(pin, user_id)
            f = Fernet(key)
            encrypted_secret_key = f.encrypt(raw_secret_key.encode()).decode()

            # Save to Firestore
            # No need to save salt as it is deterministic
            db.collection('vanity_jobs').document(job_id).update({
                'status': 'COMPLETED',
                'public_key': public_key,
                'secret_key': encrypted_secret_key,
                'completed_at': firestore.SERVER_TIMESTAMP
            })
            print(f"SECRET KEY SAVED for {job_id}")

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
        if doc.exists and doc.to_dict().get('pin_hash'):
            return jsonify({'has_pin': True})
        else:
            return jsonify({'has_pin': False})
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

    # Input Validation
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
    if not prefix and not suffix:
        return jsonify({'error': 'Prefix or suffix required'}), 400
    if not pin:
        return jsonify({'error': 'PIN is mandatory'}), 400

    if (prefix and not is_base58(prefix)) or (suffix and not is_base58(suffix)):
        return jsonify({'error': 'Invalid characters. Base58 characters only.'}), 400

    # Verify PIN first (Optional but good for security, though strict verification requires checking hash)
    # The requirement says "Accept pin... Use the pin to derive Fernet Key".
    # It doesn't explicitly ask to verify the PIN against the hash here, but it's good practice.
    # However, if the PIN is wrong here, the key will be encrypted with the wrong PIN, and the user won't be able to decrypt it later.
    # So we SHOULD verify the PIN against the stored hash.
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
        active_jobs = db.collection('vanity_jobs').where('user_id', '==', user_id).where('status', 'in', ['QUEUED', 'RUNNING']).get()
        if len(list(active_jobs)) > 0:
             return jsonify({'error': 'You can only run one job at a time.'}), 403

    except Exception as e:
         return jsonify({'error': str(e)}), 500

    # Calculate difficulty to determine priority
    total_len = len(prefix or '') + len(suffix or '')
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
            'created_at': firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Spawn Background Process
    # We spawn it immediately. It will block inside background_grinder if semaphores are full.
    p = multiprocessing.Process(target=background_grinder, args=(job_id, user_id, prefix, suffix, case_sensitive, pin, is_quick))
    p.start()

    return jsonify({'job_id': job_id, 'message': 'Job accepted'}), 202

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
            # If decryption fails (e.g. key rotation or logic error, though PIN check passed), handle it.
            # Since we verify the PIN hash first, this is unlikely unless the salt/logic changed.
            # But if the user somehow changed their PIN *after* starting the job but *before* revealing...
            # Wait, if they change their PIN, the stored hash changes.
            # But the job was encrypted with the OLD PIN.
            # If the PIN changes, old jobs become inaccessible unless we re-encrypt them.
            # The prompt doesn't ask for PIN change logic. I'll assume PIN stays same.
            return jsonify({'error': 'Decryption failed. Did you change your PIN?'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Reap zombie processes automatically
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    # Bind to 0.0.0.0 on port 80 as requested
    app.run(host='0.0.0.0', port=80)
