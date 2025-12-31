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
        endpoint = f"{server_url.rstrip('/')}/api/worker/complete"
        # Ensure we don't double-append if the env var already has the path (rare but possible)
        if server_url.endswith('/api/worker/complete'):
             endpoint = server_url

        logging.info(f"Reporting status to {endpoint}")
        requests.post(endpoint, json=payload, timeout=10).raise_for_status()
        logging.info("Status reported successfully.")
    except Exception as e:
        logging.error(f"Failed to report status: {e}")

def main():
    # 1. Parse Args manually (Cloud Run pass-through)
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

    # 2. Load Environment Variables
    job_id = os.environ.get('TASK_JOB_ID')
    server_url = os.environ.get('SERVER_URL')
    task_pin = os.environ.get('TASK_PIN')
    task_user_id = os.environ.get('TASK_USER_ID')

    # Worker Config (Set by Dockerfile)
    worker_type = os.environ.get('WORKER_TYPE', 'CPU') # Default to CPU
    binary_path = os.environ.get('BINARY_PATH', './cpu-grinder-bin')

    if not all([job_id, task_pin, task_user_id]):
        logging.error("Missing required env vars: TASK_JOB_ID, TASK_PIN, TASK_USER_ID")
        sys.exit(1)

    logging.info(f"Starting Worker. Type: {worker_type}, Binary: {binary_path}")
    logging.info(f"Job: {job_id}, Target: {prefix}...{suffix}")

    # 3. Build Command
    if not os.path.exists(binary_path):
        logging.error(f"Binary not found at {binary_path}")
        report_status(server_url, {"job_id": job_id, "error": f"Binary missing: {binary_path}"})
        sys.exit(1)

    cmd = [binary_path]
    if prefix: cmd.extend(["--prefix", prefix])
    if suffix: cmd.extend(["--suffix", suffix])
    cmd.extend(["--case-sensitive", case_sensitive])

    # 4. Run Process
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        found_pubkey = None
        found_secret = None

        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line: continue

            # Log stats (filtered)
            if line.startswith("[STATS]") or "MH/s" in line:
                print(line)
            # Check for Match
            # Format 1 (GPU): MATCH:<PUBKEY>:<PRIVKEY>
            # Format 2 (CPU): MATCH:<PUBKEY>:<PRIVKEY> (Unified now)
            elif line.startswith("MATCH:"):
                parts = line.split(':')
                if len(parts) >= 3:
                    found_pubkey = parts[1]
                    found_secret = parts[2] # This is the raw keypair (scalar+pub or seed+pub)
                    logging.info(f"Match found! Public Key: {found_pubkey}")
                    process.terminate()
                    break
            # Fallback for old CPU format "MATCH_FOUND:{B58}"
            elif line.startswith("MATCH_FOUND:"):
                full_key = line.split(':')[1]
                # We need to extract pubkey from the full keypair
                try:
                    raw = base58.b58decode(full_key)
                    # Solana/Ed25519: Last 32 bytes are public key
                    if len(raw) == 64:
                        pub_bytes = raw[32:64]
                        found_pubkey = base58.b58encode(pub_bytes).decode()
                        found_secret = full_key
                        logging.info(f"Match found (Legacy)! Public Key: {found_pubkey}")
                        process.terminate()
                        break
                except Exception as e:
                    logging.error(f"Error parsing legacy match: {e}")
            else:
                # Log other interesting lines, but keep it quiet
                if "Error" in line or "Warning" in line or "STATUS" in line:
                    logging.info(f"Binary: {line}")

        process.wait()

        if found_pubkey and found_secret:
            # 5. Encrypt Result
            logging.info("Encrypting result...")
            key = generate_key_from_pin(task_pin, task_user_id)
            encrypted_secret = Fernet(key).encrypt(found_secret.encode()).decode()

            result = {
                "job_id": job_id,
                "public_key": found_pubkey,
                "secret_key": encrypted_secret
            }
            report_status(server_url, result)
        else:
            logging.error("Process exited without finding a match.")
            report_status(server_url, {"job_id": job_id, "error": "Search failed or exited early"})
            sys.exit(1)

    except Exception as e:
        logging.exception("Worker execution failed")
        report_status(server_url, {"job_id": job_id, "error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    main()
