import subprocess
import json
import os
import glob
import sys
import base58
import argparse
import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler

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

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='CPU Grinder Worker')
    parser.add_argument('--prefix', type=str, help='Prefix to search for')
    parser.add_argument('--suffix', type=str, help='Suffix to search for')
    parser.add_argument('--case-sensitive', type=str, choices=['true', 'false'], default='true', help='Case sensitivity')

    # Cloud Run Jobs might pass arguments via sys.argv in a way that argparse handles naturally.
    # If the entrypoint is `python3 worker.py` and arguments are appended, `sys.argv` will be populated correctly.
    args = parser.parse_args()

    logging.info(f"Starting job with args: {args}")

    prefix = args.prefix
    suffix = args.suffix

    grind_args = []
    if prefix:
        grind_args.extend(["--starts-with", f"{prefix}:1"])
    elif suffix:
        grind_args.extend(["--ends-with", f"{suffix}:1"])
    else:
        error_msg = "No pattern provided"
        logging.error(error_msg)
        print(json.dumps({"found": False, "error": error_msg}))
        sys.exit(1)

    # Note: solana-keygen grind is case-sensitive by default.
    # If case-insensitive is requested, we might need to handle it.
    # However, existing logic only supported case-sensitive matching implicitly or explicitly.
    # The --case-sensitive flag is present in the arguments passed by the dispatcher.
    if args.case_sensitive == 'false':
        grind_args.append("--ignore-case")

    # Use full path to ensure reliability, or fallback to PATH
    solana_keygen_path = "/root/.local/share/solana/install/active_release/bin/solana-keygen"
    if not os.path.exists(solana_keygen_path):
        solana_keygen_path = "solana-keygen"
        logging.info("Using solana-keygen from PATH")
    else:
        logging.info(f"Using solana-keygen at {solana_keygen_path}")

    command = [solana_keygen_path, "grind"] + grind_args

    logging.info(f"Running command: {command}")

    try:
        # Run the grind
        process = subprocess.run(command, capture_output=True, text=True, check=True)

        # Log stdout/stderr for debugging (be careful with secrets, but grind usually just outputs status)
        logging.info(f"Command stdout: {process.stdout}")

        # Find the .json file solana-keygen just made
        json_files = glob.glob("*.json")
        if not json_files:
            error_msg = "Keypair file not generated"
            logging.error(error_msg)
            print(json.dumps({"found": False, "error": error_msg}))
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

        result = {
            "found": True,
            "public_key": public_key_b58,
            "private_key": private_key_b58
        }

        # Print JSON to stdout for the caller to parse
        print(json.dumps(result))

        # Clean up
        os.remove(found_file)
        logging.info("Job completed successfully")

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess error: {e.stderr}")
        print(json.dumps({"found": False, "error": str(e)}))
        sys.exit(1)
    except Exception as e:
        logging.exception("Unexpected error")
        print(json.dumps({"found": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
