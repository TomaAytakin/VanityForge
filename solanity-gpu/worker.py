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
    """
    Returns: (address, secret, fatal_error)
    fatal_error is True if we should NOT retry (e.g. Invariant Failure)
    """
    # New Rust binary path
    binary_path = "solana-vanity"

    cmd = [binary_path]
    cmd.append("--gpu")

    if prefix:
        cmd.extend(["--starts-with", prefix])
    if suffix:
        cmd.extend(["--ends-with", suffix])

    logging.info(f"Starting Rust GPU grinder with command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        found_address = None
        found_secret_str = None

        for line in process.stdout:
            line_str = line.strip()
            if not line_str:
                continue

            # Print output for debugging/user visibility
            current_time = time.strftime("%H:%M:%S", time.localtime())
            # Filter out progress bars if they clutter logs excessively,
            # but usually it's fine or they are on stderr.
            # We merged stderr, so we see everything.
            print(f"[{current_time}] ⛏️ GRINDER: {line_str}", flush=True)

            # Parse Output
            # Expected format example:
            # Address: 8...
            # Secret Key: [1, 2, ...] OR Secret Key: 4... (Base58)

            if "Address:" in line_str:
                parts = line_str.split("Address:")
                if len(parts) > 1:
                    found_address = parts[1].strip()

            if "Secret Key:" in line_str:
                parts = line_str.split("Secret Key:")
                if len(parts) > 1:
                    found_secret_str = parts[1].strip()

            # Check if we have both
            if found_address and found_secret_str:
                logging.info(f"⛏️ GRINDER: KEY FOUND!")

                try:
                    # Parse secret key
                    secret_bytes = None
                    if found_secret_str.startswith("[") and found_secret_str.endswith("]"):
                        # It's a JSON array
                        try:
                            int_list = json.loads(found_secret_str)
                            secret_bytes = bytes(int_list)
                        except json.JSONDecodeError:
                            logging.error("Failed to parse secret key as JSON array")
                    else:
                        # Assume Base58 string
                        try:
                            secret_bytes = base58.b58decode(found_secret_str)
                        except Exception:
                            logging.error("Failed to decode secret key as Base58")

                    if not secret_bytes or len(secret_bytes) != 64:
                        logging.error(f"Invalid secret key length: {len(secret_bytes) if secret_bytes else 0}")
                        # Reset and continue looking? Or abort?
                        # Usually if we found one, we are done.
                        # But if parsing failed, maybe we parsed the wrong line.
                        # For safety, let's reset and continue if verification fails,
                        # or terminate if we are sure it's the result.
                        # But usually the tool exits after finding one.
                        pass

                    else:
                        # CRITICAL: Verify Ed25519 Invariant
                        # The secret key is full 64 bytes (private + public)
                        priv_bytes = secret_bytes[:32]
                        priv_key = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
                        derived_pub_bytes = priv_key.public_key().public_bytes(
                            encoding=serialization.Encoding.Raw,
                            format=serialization.PublicFormat.Raw
                        )
                        derived_pub_b58 = base58.b58encode(derived_pub_bytes).decode('ascii')

                        if derived_pub_b58 != found_address:
                            logging.error("CRITICAL: Generated key FAILS Ed25519 invariant! Discarding.")
                            logging.error(f"Provided Pub: {found_address}")
                            logging.error(f"Derived Pub:  {derived_pub_b58}")

                            process.terminate()
                            return None, None, True # Fatal

                        # All good
                        secret_b58 = base58.b58encode(secret_bytes).decode('ascii')

                        process.terminate()
                        return found_address, secret_b58, False

                except Exception as inv_err:
                    logging.error(f"Invariant check raised exception: {inv_err}")
                    process.terminate()
                    return None, None, True

        process.wait()

        if process.returncode != 0:
            logging.error(f"Binary Error: Return Code {process.returncode}")

        logging.error("Binary finished but no valid keypair found.")
        return None, None, False

    except Exception as e:
        logging.exception(f"Unexpected error in run_gpu_grinder: {e}")
        return None, None, False

if __name__ == "__main__":
    print("🩺 DIAGNOSTIC: Checking GPU Health...")
    # Skip chmod for new binary as it is built inside container usually with correct perms
    # but doesn't hurt to try if we knew the path.
    # Since path is dynamic/target dir, we skip chmod or do it if we find it.

    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        logging.error("nvidia-smi not found. Ensure the container has GPU access.")
    except Exception as e:
        logging.error(f"Error running nvidia-smi: {e}")

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
            address, secret, fatal = run_gpu_grinder(TASK_PREFIX, TASK_SUFFIX)
            if address and secret:
                break

            if fatal:
                logging.error("Fatal error encountered (Invariant Failure). Aborting retries.")
                break

            logging.warning(f"Grinding attempt {i+1}/{MAX_RETRIES} failed. Retrying...")
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
