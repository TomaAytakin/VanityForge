import os
import uuid
import logging
import multiprocessing
import base58
import re
import time
import signal
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import firestore
from solders.keypair import Keypair

# Configuration
PROJECT_ID = 'vanityforge'

# Serve static files from the current directory
app = Flask(__name__, static_folder='.', static_url_path='')
# Strict CORS: Only allow requests from the specific frontend URL
CORS(app, resources={r"/submit-job": {"origins": "https://tomaaytakin.github.io"}})

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Global to track active jobs
# Note: In a production server like Gunicorn, this might not work as expected if multiple workers are used for Flask itself.
# But for a simple `python vm_server.py` (single threaded flask or threaded), this variable is local to the process.
# We will use a Semaphore to limit concurrent background processes.
# Limiting to 2 concurrent jobs to prevent overloading the 4-core expectation (total 8+ processes if 2 jobs).
# Actually, the user says "Initialize multiprocessing.Pool with exactly 4 processes".
# If we have 1 job running, we use 4 processes.
# If we allow more, we might oversubscribe.
# Let's limit to 1 concurrent job for safety given "Cost optimization".
MAX_CONCURRENT_JOBS = 1
active_jobs_semaphore = multiprocessing.Semaphore(MAX_CONCURRENT_JOBS)

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

def background_grinder(job_id, prefix, suffix, case_sensitive):
    """
    Background process that manages the Pool of workers.
    """
    try:
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

            public_key, secret_key = found_key

            # Save to Firestore
            db.collection('vanity_jobs').document(job_id).update({
                'status': 'COMPLETED',
                'public_key': public_key,
                'secret_key': secret_key,
                'completed_at': firestore.SERVER_TIMESTAMP
            })

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
        active_jobs_semaphore.release()

@app.route('/submit-job', methods=['POST'])
def submit_job():
    # Options handled by flask-cors automatically

    data = request.get_json(silent=True) or {}
    user_id = data.get('userId') or data.get('user_id')
    prefix = data.get('prefix')
    suffix = data.get('suffix', '')
    case_sensitive = data.get('case_sensitive', True)

    # Input Validation
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
    if not prefix and not suffix:
        return jsonify({'error': 'Prefix or suffix required'}), 400

    if (prefix and not is_base58(prefix)) or (suffix and not is_base58(suffix)):
        return jsonify({'error': 'Invalid characters. Base58 characters only.'}), 400

    # Check concurrency
    if not active_jobs_semaphore.acquire(block=False):
         return jsonify({'error': 'Server busy. Please try again later.'}), 503

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
        active_jobs_semaphore.release()
        return jsonify({'error': str(e)}), 500

    # Spawn Background Process
    p = multiprocessing.Process(target=background_grinder, args=(job_id, prefix, suffix, case_sensitive))
    p.start()

    return jsonify({'job_id': job_id, 'message': 'Job accepted'}), 202

if __name__ == '__main__':
    # Reap zombie processes automatically
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    # Bind to 0.0.0.0 on port 80 as requested
    app.run(host='0.0.0.0', port=80)
