import subprocess
import json
import os
import glob
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

def setup_logging():
    """Sets up Google Cloud Logging if available, otherwise falls back to standard logging."""
    try:
        # Instantiates a client
        client = google.cloud.logging.Client()
        handler = CloudLoggingHandler(client)
        google.cloud.logging.handlers.setup_logging(handler)
        logging.getLogger().setLevel(logging.INFO)
    except Exception as e:
        # Fallback to standard logging if credentials fail (e.g. local test)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.warning(f"Failed to setup Cloud Logging: {e}. using basic logging.")

def get_deterministic_salt(user_id):
    return hashlib.sha256(user_id.encode()).digest()

def generate_key_from_pin(pin, user_id):
    salt = get_deterministic_salt(user_id)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))

def report_status(server_url, payload):
    """Sends the result or error back to the server."""
    if not server_url:
        logging.error("SERVER_URL not set, cannot report status.")
        return

    try:
        # Ensure url ends with /api/worker/complete or append it if it's just base url
        # The env var is likely just the base url "https://vanityforge.org"
        if not server_url.endswith('/api/worker/complete'):
            endpoint = f"{server_url.rstrip('/')}/api/worker/complete"
        else:
            endpoint = server_url

        logging.info(f"Reporting status to {endpoint}")
        response = requests.post(endpoint, json=payload, timeout=10)
        response.raise_for_status()
        logging.info("Successfully reported status to server.")
    except Exception as e:
        logging.error(f"Failed to report status to server: {e}")

def main():
    setup_logging()
    
    # Manual argument parsing to be more robust on Cloud Run
    prefix = None
    suffix = None
    case_sensitive = 'true'

    args = sys.argv[1:]
    logging.info(f"Raw arguments: {args}")

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--prefix':
            if i + 1 < len(args):
                prefix = args[i+1]
                i += 2
            else:
                logging.error("--prefix requires an argument")
                sys.exit(1)
        elif arg.startswith('--prefix='):
            prefix = arg.split('=', 1)[1]
            i += 1
        elif arg == '--suffix':
            if i + 1 < len(args):
                suffix = args[i+1]
                i += 2
            else:
                logging.error("--suffix requires an argument")
                sys.exit(1)
        elif arg.startswith('--suffix='):
            suffix = arg.split('=', 1)[1]
            i += 1
        elif arg == '--case-sensitive':
            if i + 1 < len(args):
                case_sensitive = args[i+1]
                i += 2
            else:
                logging.error("--case-sensitive requires an argument")
                sys.exit(1)
        elif arg.startswith('--case-sensitive='):
            case_sensitive = arg.split('=', 1)[1]
            i += 1
        else:
            # Ignore unknown args or handle as needed
            i += 1

    job_id = os.environ.get('TASK_JOB_ID')
    server_url = os.environ.get('SERVER_URL')

    if not job_id:
        logging.warning("TASK_JOB_ID not set. Proceeding but callback might fail.")

    grind_args = []
    if prefix:
        grind_args.extend(["--starts-with", f"{prefix}:1"])
    elif suffix:
        grind_args.extend(["--ends-with", f"{suffix}:1"])
    else:
        error_msg = "No pattern provided (prefix or suffix required)"
        logging.error(error_msg)
        report_status(server_url, {"job_id": job_id, "error": error_msg})
        sys.exit(1)

    if case_sensitive == 'false':
        grind_args.append("--ignore-case")

    # Check for Turbo Binary
    gpu_binary = "./gpu_grinder"
    use_turbo = os.path.exists(gpu_binary)

    if use_turbo:
        logging.info("🚀 Starting L4 Turbo Grinder...")
        # Reconstruct args for C++ binary
        command = [gpu_binary]
        if prefix: command.extend(["--prefix", prefix])
        if suffix: command.extend(["--suffix", suffix])
        if case_sensitive == 'true': command.extend(["--case-sensitive", "true"])

        logging.info(f"Running Turbo command: {command}")

        start_time = time.time()

        # Live Streaming Logic
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        # Read stdout line by line
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line: continue

            # Passthrough standardized tags
            if line.startswith("[STATS]") or line.startswith("[STATUS]") or line.startswith("[ERROR]") or line.startswith("[SUCCESS]"):
                print(line, flush=True) # Direct to Cloud Run logs
                logging.info(line)
            else:
                logging.debug(f"Worker output: {line}")

        process.wait()

        if process.returncode != 0:
            logging.error(f"Turbo Grinder failed with code {process.returncode}")
            # Fallback or exit?
            sys.exit(process.returncode)

    else:
        # Fallback to Legacy Rust Grinder
        solana_keygen_cmd = "/usr/local/bin/solana-keygen"
        command = [solana_keygen_cmd, "grind"] + grind_args

        logging.info(f"Running command: {command}")

        start_time = time.time()
        # Run the grind
        process = subprocess.run(command, capture_output=True, text=True, check=True)

        # Legacy Hashrate Calc (Post-Run)
        end_time = time.time()
        elapsed = end_time - start_time
        if elapsed <= 0: elapsed = 0.001
        length = len(prefix) if prefix else len(suffix)
        hashrate_mhs = (58**length) / elapsed / 1_000_000
        logging.info(f"Hashrate: {hashrate_mhs:.1f} MH/s")

        # Find the .json file solana-keygen just made
        json_files = glob.glob("*.json")
        if not json_files:
            error_msg = "Keypair file not generated"
            logging.error(error_msg)
            report_status(server_url, {"job_id": job_id, "error": error_msg})
            sys.exit(1)

        found_file = json_files[0]
        logging.info(f"Found keypair file: {found_file}")

        with open(found_file, "r") as f:
            keypair_data = json.load(f)

        private_key_bytes = bytes(keypair_data)
        private_key_b58 = base58.b58encode(private_key_bytes).decode('utf-8')
        
        # Last 32 bytes are always the public key in a 64-byte Solana keypair
        public_key_bytes = private_key_bytes[32:]
        public_key_b58 = base58.b58encode(public_key_bytes).decode('utf-8')

        # V19 Security Update: Internal Encryption
        # Sync with Environment
        task_pin = os.environ.get('TASK_PIN')
        task_user_id = os.environ.get('TASK_USER_ID')

        if not task_pin or not task_user_id:
            error_msg = "V19 Security Error: Missing TASK_PIN or TASK_USER_ID env vars"
            logging.error(error_msg)
            report_status(server_url, {"job_id": job_id, "error": error_msg})
            sys.exit(1)

        # Encrypt
        key = generate_key_from_pin(task_pin, task_user_id)
        encrypted_secret_key = Fernet(key).encrypt(private_key_b58.encode()).decode()

        result = {
            "job_id": job_id,
            "public_key": public_key_b58,
            "secret_key": encrypted_secret_key
        }

        # Send Callback
        report_status(server_url, result)

        # Clean up
        os.remove(found_file)
        logging.info("Job completed successfully")

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess error: {e.stderr}")
        report_status(server_url, {"job_id": job_id, "error": str(e)})
        sys.exit(1)
    except FileNotFoundError:
        logging.error("solana-keygen not found in PATH")
        report_status(server_url, {"job_id": job_id, "error": "solana-keygen executable not found"})
        sys.exit(1)
    except Exception as e:
        logging.exception("Unexpected error")
        report_status(server_url, {"job_id": job_id, "error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    main()
