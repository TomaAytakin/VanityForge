import subprocess
import os
import json
import logging
import base64
import hashlib
import sys
import time
import base58
import requests
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def get_deterministic_salt(user_id):
    return hashlib.sha256(user_id.encode()).digest()

def generate_key_from_pin(pin, user_id):
    salt = get_deterministic_salt(user_id)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))

def run_gpu_grinder(prefix, suffix, gpu_index=0):
    cmd = ["./solanity"]

    # Updated Logic: Only pass prefix, binary handles range calc if needed or just scans
    # The C++ binary was refactored to handle --prefix-str directly for filtering
    cmd.extend(["--prefix-str", prefix or ""])

    if suffix:
        cmd.extend(["--suffix", suffix])

    cmd.extend(["--gpu-index", str(gpu_index)])

    logging.info(f"Starting GPU grinder with command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        final_json_str = ""
        found_json = False

        for line in process.stdout:
            line_str = line.strip()
            if not line_str:
                continue

            if '"public_key":' in line_str and '"secret_key":' in line_str:
                final_json_str = line_str
                found_json = True
                logging.info(f"⛏️ GRINDER: KEY FOUND!")
                process.terminate()
                break
            else:
                current_time = time.strftime("%H:%M:%S", time.localtime())
                print(f"[{current_time}] ⛏️ GRINDER: {line_str}", flush=True)

        process.wait()

        if not found_json and process.returncode != 0:
            logging.error(f"Binary Error: Return Code {process.returncode}")
            return None, None

        if not found_json or not final_json_str:
            logging.error("Binary finished but no JSON output found.")
            return None, None

        try:
            data = json.loads(final_json_str)
        except json.JSONDecodeError as e:
             logging.error(f"Error parsing GPU output: {e}")
             # REDACTED RAW OUTPUT FOR SECURITY
             logging.error("Raw output redacted due to potential sensitive content.")
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
    print("🩺 DIAGNOSTIC: Checking GPU Health...")
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        logging.error("nvidia-smi not found. Ensure the container has GPU access.")
    except Exception as e:
        logging.error(f"Error running nvidia-smi: {e}")

    # No nvcc check needed as we are in python runtime without nvcc likely

    TASK_JOB_ID = os.environ.get("TASK_JOB_ID")
    TASK_PREFIX = os.environ.get("TASK_PREFIX", "")
    TASK_SUFFIX = os.environ.get("TASK_SUFFIX", "")
    TASK_PIN = os.environ.get("TASK_PIN")
    TASK_USER_ID = os.environ.get("TASK_USER_ID")
    SERVER_URL = os.environ.get("SERVER_URL")

    if not TASK_JOB_ID or not TASK_PIN or not TASK_USER_ID or not SERVER_URL:
        logging.error("Missing required environment variables: TASK_JOB_ID, TASK_PIN, TASK_USER_ID, SERVER_URL")
        sys.exit(1)

    logging.info(f"Worker started for Job ID: {TASK_JOB_ID}")

    try:
        # No more Firebase DB initialization

        # Remove min/max limit usage as requested
        address, secret = run_gpu_grinder(TASK_PREFIX, TASK_SUFFIX)

        if address and secret:
            key = generate_key_from_pin(TASK_PIN, TASK_USER_ID)
            enc_key = Fernet(key).encrypt(secret.encode()).decode()

            success = False
            for attempt in range(5):
                try:
                    payload = {
                        'job_id': TASK_JOB_ID,
                        'public_key': address,
                        'secret_key': enc_key
                    }
                    resp = requests.post(f"{SERVER_URL}/api/worker/complete", json=payload, timeout=10)

                    if resp.status_code == 200:
                        logging.info("INFO: SERVER_UPDATE_SUCCESS")
                        logging.info(f"Job {TASK_JOB_ID} reported successfully. Address: {address}")
                        success = True
                        break
                    else:
                        logging.warning(f"Warning: Server update failed (status {resp.status_code}): {resp.text}")
                        time.sleep(2 ** attempt)

                except Exception as update_err:
                    logging.warning(f"Warning: Server connection failed (attempt {attempt+1}/5): {update_err}")
                    time.sleep(2 ** attempt)

            if not success:
                logging.error("ERROR: SERVER_UPDATE_FAILED")
                sys.exit(1)

            sys.exit(0)

        else:
            logging.error("Failed to generate address.")
            # Report failure to server
            try:
                requests.post(f"{SERVER_URL}/api/worker/complete", json={
                    'job_id': TASK_JOB_ID,
                    'error': 'GPU Worker failed to generate address'
                }, timeout=10)
            except:
                pass
            sys.exit(1)

    except Exception as e:
        logging.exception(f"Worker failed: {e}")
        try:
            requests.post(f"{SERVER_URL}/api/worker/complete", json={
                'job_id': TASK_JOB_ID,
                'error': str(e)
            }, timeout=10)
        except:
            pass
        sys.exit(1)
