import subprocess
import os
import json
import logging
import firebase_admin
from firebase_admin import credentials, firestore
import base64
import hashlib
import sys
import time
import base58
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Initialize Firebase
# Note: In Cloud Run, credentials are often auto-detected or passed via env var.
# We use ApplicationDefault.
cred = credentials.ApplicationDefault()
try:
    firebase_admin.initialize_app(cred)
except ValueError:
    # Already initialized
    pass

db = firestore.client()

def get_deterministic_salt(user_id):
    return hashlib.sha256(user_id.encode()).digest()

def generate_key_from_pin(pin, user_id):
    salt = get_deterministic_salt(user_id)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))

def run_gpu_grinder(prefix, suffix, gpu_index=0):
    # Call the solanity binary
    # usage: ./solanity --prefix <p> --suffix <s> --gpu-index <idx>
    # Note: We do not add a JSON flag because the binary outputs JSON by default on success.
    cmd = ["./solanity"]

    if prefix:
        cmd.extend(["--prefix", prefix])
    if suffix:
        cmd.extend(["--suffix", suffix])

    cmd.extend(["--gpu-index", str(gpu_index)])

    logging.info(f"Starting GPU grinder with command: {' '.join(cmd)}")

    try:
        # Run the command and capture output
        # The binary prints JSON to stdout and exits with 0 on success
        # check=False to handle return code manually
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            logging.error(f"Binary Error: Return Code {result.returncode}")
            logging.error(f"Stderr: {stderr}")
            return None, None

        if not stdout:
            logging.error("Binary Error: Empty stdout")
            logging.error(f"Stderr: {stderr}")
            return None, None

        # The stdout might contain other logs if not careful, but main.cu seems clean.
        # Find the JSON part if there's noise?
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            # Fallback: try to find the start and end of JSON
            start = stdout.find('{')
            end = stdout.rfind('}')
            if start != -1 and end != -1:
                try:
                    json_str = stdout[start:end+1]
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    logging.error(f"Error parsing GPU output: {e}")
                    logging.error(f"Raw output: {stdout}")
                    logging.error(f"Stderr: {stderr}")
                    return None, None
            else:
                logging.error(f"Error parsing GPU output: {e}")
                logging.error(f"Raw output: {stdout}")
                logging.error(f"Stderr: {stderr}")
                return None, None

        address = data.get("public_key")
        secret_ints = data.get("secret_key")

        secret_b58 = None
        if secret_ints:
            secret_bytes = bytes(secret_ints)
            secret_b58 = base58.b58encode(secret_bytes).decode('ascii')

        return address, secret_b58

    except Exception as e:
        logging.exception(f"Unexpected error in run_gpu_grinder: {e}")
        return None, None

if __name__ == "__main__":
    TASK_JOB_ID = os.environ.get("TASK_JOB_ID")
    TASK_PREFIX = os.environ.get("TASK_PREFIX", "")
    TASK_SUFFIX = os.environ.get("TASK_SUFFIX", "")
    # TASK_CASE not used in main.cu?
    # Checking main.cu: it only takes prefix/suffix. It does case-sensitive check logic inside?
    # main.cu:
    # if (key[j] != prefix.data[j]) -> looks case sensitive.
    # main.cu does NOT have --ignore-case flag implemented in argv parsing.
    # Assuming case sensitive is always enforced or handled by prefix/suffix casing passed in.

    TASK_PIN = os.environ.get("TASK_PIN")
    TASK_USER_ID = os.environ.get("TASK_USER_ID")

    if not TASK_JOB_ID or not TASK_PIN or not TASK_USER_ID:
        logging.error("Missing required environment variables: TASK_JOB_ID, TASK_PIN, TASK_USER_ID")
        sys.exit(1)

    logging.info(f"Worker started for Job ID: {TASK_JOB_ID}")

    try:
        # Update Firestore job status to 'RUNNING' (if not already)
        job_ref = db.collection('vanity_jobs').document(TASK_JOB_ID)
        # We might not need to set running here if VM server did it, but it's safe.
        # However, vm_server.py does set it to RUNNING before dispatching?
        # Yes: db.collection('vanity_jobs').document(job_id).update({'status': 'RUNNING'})

        # Call grinder
        address, secret = run_gpu_grinder(TASK_PREFIX, TASK_SUFFIX)

        if address and secret:
            # Encrypt the resulting private key
            key = generate_key_from_pin(TASK_PIN, TASK_USER_ID)
            enc_key = Fernet(key).encrypt(secret.encode()).decode()

            # Retry loop for Firestore update
            success = False
            for attempt in range(5):
                try:
                    # Update Firestore job status to 'COMPLETED' with the result
                    job_ref.update({
                        'status': 'COMPLETED',
                        'public_key': address,
                        'secret_key': enc_key,
                        'completed_at': firestore.SERVER_TIMESTAMP
                    })
                    # Force a synchronous blocking read to ensure data is committed before exiting
                    job_ref.get()
                    logging.info("INFO: FOUND_KEY_UPDATE_SUCCESS")
                    logging.info(f"Job {TASK_JOB_ID} completed successfully. Address: {address}")
                    success = True
                    break
                except Exception as update_err:
                    logging.warning(f"Warning: Firestore update failed (attempt {attempt+1}/5): {update_err}")
                    time.sleep(2 ** attempt) # Exponential backoff

            if not success:
                logging.error("ERROR: FIRESTORE_UPDATE_FAILED")
                sys.exit(1)

        else:
            logging.error("Failed to generate address.")
            job_ref.update({'status': 'FAILED', 'error': 'GPU Worker failed to generate address'})
            sys.exit(1)

    except Exception as e:
        logging.exception(f"Worker failed: {e}")
        try:
            db.collection('vanity_jobs').document(TASK_JOB_ID).update({'status': 'FAILED', 'error': str(e)})
        except:
            pass
        sys.exit(1)
