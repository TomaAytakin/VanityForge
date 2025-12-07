import subprocess
import os
import firebase_admin
from firebase_admin import credentials, firestore
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Initialize Firebase
cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred)
db = firestore.client()

def get_deterministic_salt(user_id):
    return hashlib.sha256(user_id.encode()).digest()

def generate_key_from_pin(pin, user_id):
    salt = get_deterministic_salt(user_id)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))

def run_rust_grinder(prefix, suffix, case_sensitive):
    cmd = ["solana-vanity", "--prefix", prefix, "--suffix", suffix, "--no-grid"]
    if not case_sensitive:
        cmd.append("--ignore-case")

    try:
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = result.stdout

        address = None
        secret = None

        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("Address:"):
                address = line.split("Address:", 1)[1].strip()
            if line.startswith("Secret:"):
                secret = line.split("Secret:", 1)[1].strip()

        return address, secret
    except subprocess.CalledProcessError as e:
        print(f"Error running rust grinder: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None

if __name__ == "__main__":
    TASK_JOB_ID = os.environ.get("TASK_JOB_ID")
    TASK_PREFIX = os.environ.get("TASK_PREFIX", "")
    TASK_SUFFIX = os.environ.get("TASK_SUFFIX", "")
    TASK_CASE = os.environ.get("TASK_CASE", "True")
    TASK_PIN = os.environ.get("TASK_PIN")
    TASK_USER_ID = os.environ.get("TASK_USER_ID")

    if not TASK_JOB_ID or not TASK_PIN or not TASK_USER_ID:
        print("Missing required environment variables: TASK_JOB_ID, TASK_PIN, TASK_USER_ID")
        exit(1)

    # Parse case sensitive boolean from string
    case_sensitive = str(TASK_CASE).lower() == 'true'

    try:
        # Update Firestore job status to 'RUNNING'
        job_ref = db.collection('vanity_jobs').document(TASK_JOB_ID)
        job_ref.update({'status': 'RUNNING'})

        # Call run_rust_grinder
        address, secret = run_rust_grinder(TASK_PREFIX, TASK_SUFFIX, case_sensitive)

        if address and secret:
            # Encrypt the resulting private key
            key = generate_key_from_pin(TASK_PIN, TASK_USER_ID)
            enc_key = Fernet(key).encrypt(secret.encode()).decode()

            # Update Firestore job status to 'COMPLETED' with the result
            job_ref.update({
                'status': 'COMPLETED',
                'public_key': address,
                'secret_key': enc_key,
                'completed_at': firestore.SERVER_TIMESTAMP
            })
            print(f"Job {TASK_JOB_ID} completed successfully. Address: {address}")
        else:
            print("Failed to generate address.")
            job_ref.update({'status': 'FAILED', 'error': 'Worker failed to generate address'})

    except Exception as e:
        print(f"Worker failed: {e}")
        try:
            db.collection('vanity_jobs').document(TASK_JOB_ID).update({'status': 'FAILED', 'error': str(e)})
        except:
            pass
