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
cred = credentials.ApplicationDefault()
try:
    firebase_admin.initialize_app(cred)
except ValueError:
    pass

db = firestore.client()

def get_deterministic_salt(user_id):
    return hashlib.sha256(user_id.encode()).digest()

def generate_key_from_pin(pin, user_id):
    salt = get_deterministic_salt(user_id)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))

def calculate_prefix_range(prefix_str):
    """
    Calculates the target u64 value and mask for the GPU filter based on the Base58 prefix.
    """
    if not prefix_str:
        return 0, 0

    # Base58 Alphabet
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

    # Calculate min: prefix + '1111...' (padding to ~44 chars, typical solana address)
    # Calculate max: prefix + 'zzzz...'
    # Typical address length is 32 bytes -> ~44 base58 chars.
    target_len = 44
    if len(prefix_str) > target_len:
        target_len = len(prefix_str) # Should not happen

    min_str = prefix_str.ljust(target_len, '1')
    max_str = prefix_str.ljust(target_len, 'z')

    try:
        min_bytes = base58.b58decode(min_str)
        max_bytes = base58.b58decode(max_str)

        # Ensure 32 bytes. b58decode might return less if leading 1s (0s).
        # We want top bytes.
        # Pad left with 0s to 32 bytes
        min_bytes = min_bytes.rjust(32, b'\0')
        max_bytes = max_bytes.rjust(32, b'\0')

        # Extract top 8 bytes (big endian for comparison)
        min_val = int.from_bytes(min_bytes[:8], byteorder='big')
        max_val = int.from_bytes(max_bytes[:8], byteorder='big')

        # Calculate Common Prefix Mask
        # XOR min and max. Bits that changed are 1.
        diff = min_val ^ max_val

        # Find highest set bit in diff.
        # All bits ABOVE that are common.
        if diff == 0:
            mask = 0xFFFFFFFFFFFFFFFF
        else:
            # Power of 2 greater than diff
            # Or scan from MSB.
            # In Python, bit_length() gives bits required to represent.
            # E.g. diff = 0b00101... -> bit_length = 3 (if 1 is at index 2).
            # We want to clear bits from bit_length down to 0.
            shift = diff.bit_length()
            # Mask is all 1s shifted left by 'shift'.
            # But wait, 64-bit mask.
            mask = (0xFFFFFFFFFFFFFFFF << shift) & 0xFFFFFFFFFFFFFFFF

        target = min_val & mask

        return target, mask

    except Exception as e:
        logging.error(f"Error calculating prefix range: {e}")
        return 0, 0

def run_gpu_grinder(prefix, suffix, gpu_index=0):
    cmd = ["./solanity"]

    prefix_val, mask_val = calculate_prefix_range(prefix)

    if mask_val == 0:
        logging.error("GPU Filter: Mask is 0 (Pass All). This is not allowed on GPU architecture. Please provide a more specific prefix.")
        return None, None

    cmd.extend(["--prefix-val", hex(prefix_val)])
    cmd.extend(["--mask-val", hex(mask_val)])
    logging.info(f"GPU Filter: Prefix={hex(prefix_val)} Mask={hex(mask_val)}")

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
            else:
                current_time = time.strftime("%H:%M:%S", time.localtime())
                print(f"[{current_time}] ⛏️ GRINDER: {line_str}", flush=True)

        process.wait()

        if process.returncode != 0:
            logging.error(f"Binary Error: Return Code {process.returncode}")
            return None, None

        if not found_json or not final_json_str:
            logging.error("Binary finished but no JSON output found.")
            return None, None

        try:
            data = json.loads(final_json_str)
        except json.JSONDecodeError as e:
             logging.error(f"Error parsing GPU output: {e}")
             logging.error(f"Raw output: {final_json_str}")
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

    try:
        subprocess.run(["nvcc", "--version"], check=False)
    except FileNotFoundError:
        logging.warning("nvcc not found. This may be normal in runtime containers.")
    except Exception as e:
        logging.error(f"Error running nvcc: {e}")

    TASK_JOB_ID = os.environ.get("TASK_JOB_ID")
    TASK_PREFIX = os.environ.get("TASK_PREFIX", "")
    TASK_SUFFIX = os.environ.get("TASK_SUFFIX", "")
    TASK_PIN = os.environ.get("TASK_PIN")
    TASK_USER_ID = os.environ.get("TASK_USER_ID")

    if not TASK_JOB_ID or not TASK_PIN or not TASK_USER_ID:
        logging.error("Missing required environment variables: TASK_JOB_ID, TASK_PIN, TASK_USER_ID")
        sys.exit(1)

    logging.info(f"Worker started for Job ID: {TASK_JOB_ID}")

    try:
        job_ref = db.collection('vanity_jobs').document(TASK_JOB_ID)
        address, secret = run_gpu_grinder(TASK_PREFIX, TASK_SUFFIX)

        if address and secret:
            key = generate_key_from_pin(TASK_PIN, TASK_USER_ID)
            enc_key = Fernet(key).encrypt(secret.encode()).decode()

            success = False
            for attempt in range(5):
                try:
                    job_ref.update({
                        'status': 'COMPLETED',
                        'public_key': address,
                        'secret_key': enc_key,
                        'completed_at': firestore.SERVER_TIMESTAMP
                    })
                    job_ref.get()
                    logging.info("INFO: FOUND_KEY_UPDATE_SUCCESS")
                    logging.info(f"Job {TASK_JOB_ID} completed successfully. Address: {address}")
                    success = True
                    break
                except Exception as update_err:
                    logging.warning(f"Warning: Firestore update failed (attempt {attempt+1}/5): {update_err}")
                    time.sleep(2 ** attempt)

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
