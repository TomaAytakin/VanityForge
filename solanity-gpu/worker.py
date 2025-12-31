import subprocess
import json
import os
import sys
import base58
import logging
import requests
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
import time
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import multiprocessing

# Configure Logging
try:
    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client)
    google.cloud.logging.handlers.setup_logging(handler)
    logging.getLogger().setLevel(logging.INFO)
except:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_deterministic_salt(user_id):
    return hashlib.sha256(user_id.encode()).digest()

def generate_key_from_pin(pin, user_id):
    salt = get_deterministic_salt(user_id)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))

def report_status(server_url, payload):
    if not server_url:
        logging.error("SERVER_URL not set")
        return
    try:
        endpoint = f"{server_url.rstrip('/')}/api/worker/complete" if not server_url.endswith('/api/worker/complete') else server_url
        requests.post(endpoint, json=payload, timeout=10).raise_for_status()
        logging.info("Status reported successfully.")
    except Exception as e:
        logging.error(f"Failed to report status: {e}")

def main():
    # Parse Args manually (Cloud Run pass-through)
    args = sys.argv[1:]
    prefix, suffix, case_sensitive = None, None, 'true'

    idx = 0
    while idx < len(args):
        if args[idx] == '--prefix' and idx+1 < len(args):
            prefix = args[idx+1]; idx+=2
        elif args[idx] == '--suffix' and idx+1 < len(args):
            suffix = args[idx+1]; idx+=2
        elif args[idx] == '--case-sensitive' and idx+1 < len(args):
            case_sensitive = args[idx+1]; idx+=2
        else:
            idx+=1

    # Env Vars
    job_id = os.environ.get('TASK_JOB_ID')
    server_url = os.environ.get('SERVER_URL')
    task_pin = os.environ.get('TASK_PIN')
    task_user_id = os.environ.get('TASK_USER_ID')

    if not all([job_id, task_pin, task_user_id]):
        logging.error("Missing required env vars: TASK_JOB_ID, TASK_PIN, TASK_USER_ID")
        sys.exit(1)

    # Build Command
    cmd = ["./gpu_grinder"]
    if prefix: cmd.extend(["--prefix", prefix])
    if suffix: cmd.extend(["--suffix", suffix])
    cmd.extend(["--case-sensitive", case_sensitive])

    logging.info(f"Starting GPU Grinder: {cmd}")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        found_pubkey = None
        found_privkey = None

        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line: continue

            # Log stats to Cloud Logging (filtered)
            if line.startswith("[STATS]"):
                print(line) # Stdout for local view
            elif line.startswith("MATCH:"):
                # Protocol: MATCH:<PUBKEY>:<PRIVKEY>
                parts = line.split(':')
                if len(parts) >= 3:
                    found_pubkey = parts[1]
                    found_privkey = parts[2]
                    logging.info("Match found by GPU process!")
                    process.terminate()
                    break
            else:
                logging.info(f"GPU: {line}")

        process.wait()

        if found_pubkey and found_privkey:
            # Encrypt Result
            key = generate_key_from_pin(task_pin, task_user_id)
            encrypted_secret = Fernet(key).encrypt(found_privkey.encode()).decode()

            result = {
                "job_id": job_id,
                "public_key": found_pubkey,
                "secret_key": encrypted_secret
            }
            report_status(server_url, result)
        else:
            logging.error("GPU process exited without finding a match.")
            report_status(server_url, {"job_id": job_id, "error": "GPU search failed or exited early"})
            sys.exit(1)

    except Exception as e:
        logging.exception("Worker failure")
        report_status(server_url, {"job_id": job_id, "error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    main()
