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
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def get_deterministic_salt(user_id):
    return hashlib.sha256(user_id.encode()).digest()

def generate_key_from_pin(pin, user_id):
    salt = get_deterministic_salt(user_id)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))

def run_gpu_grinder(prefix, suffix):
    cmd = ["./solanity"]
    if prefix:
        cmd.extend(["--prefix", prefix])
    if suffix:
        cmd.extend(["--suffix", suffix])

    logging.info(f"Starting GPU grinder with command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            line_str = line.strip()
            if not line_str:
                continue

            try:
                if '"public_key":' in line_str:
                    data = json.loads(line_str)
                    if "public_key" in data:
                        logging.info(f"⛏️ GRINDER: KEY FOUND!")

                        address = data.get("public_key")
                        secret_ints = data.get("secret_key")

                        if secret_ints and len(secret_ints) == 64:
                            secret_bytes = bytes(secret_ints)
                            seed = secret_bytes[:32]
                            provided_pub_bytes = secret_bytes[32:]

                            # CRITICAL: Verify Ed25519 Invariant
                            try:
                                priv_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
                                derived_pub_bytes = priv_key.public_key().public_bytes(
                                    encoding=serialization.Encoding.Raw,
                                    format=serialization.PublicFormat.Raw
                                )

                                if derived_pub_bytes != provided_pub_bytes:
                                    logging.error("CRITICAL: Generated key FAILS Ed25519 invariant! Discarding.")
                                    # We do NOT terminate process here if we want to keep searching,
                                    # but the GPU binary usually exits on find.
                                    # If the GPU binary exited, we are out of luck for this run.
                                    # We should treat this as a failure and let the loop handle it?
                                    # But wait, run_gpu_grinder is called once.
                                    # If we terminate, we return None.
                                    process.terminate()
                                    return None, None

                            except Exception as inv_err:
                                logging.error(f"Invariant check raised exception: {inv_err}")
                                process.terminate()
                                return None, None

                            process.terminate()
                            secret_b58 = base58.b58encode(secret_bytes).decode('ascii')
                            return address, secret_b58
                        else:
                            logging.error("Invalid secret key length.")

            except json.JSONDecodeError:
                pass
            except Exception as e:
                logging.error(f"Error parsing line: {e}")

            current_time = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{current_time}] ⛏️ GRINDER: {line_str}", flush=True)

        process.wait()

        if process.returncode != 0:
            logging.error(f"Binary Error: Return Code {process.returncode}")

        logging.error("Binary finished but no JSON output found.")
        return None, None

    except Exception as e:
        logging.exception(f"Unexpected error in run_gpu_grinder: {e}")
        return None, None

if __name__ == "__main__":
    print("🩺 DIAGNOSTIC: Checking GPU Health...")
    try:
        subprocess.run(["chmod", "+x", "./solanity"], check=False)
    except Exception as e:
        logging.error(f"Error chmoding binary: {e}")

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
        # Retry loop for grinding
        MAX_RETRIES = 5
        address = None
        secret = None

        for i in range(MAX_RETRIES):
            address, secret = run_gpu_grinder(TASK_PREFIX, TASK_SUFFIX)
            if address and secret:
                break
            logging.warning(f"Grinding attempt {i+1}/{MAX_RETRIES} failed or produced invalid key. Retrying...")
            time.sleep(1)

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
            logging.error("Failed to generate address after retries.")
            # Report failure to server
            try:
                requests.post(f"{SERVER_URL}/api/worker/complete", json={
                    'job_id': TASK_JOB_ID,
                    'error': 'GPU Worker failed to generate valid address'
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
