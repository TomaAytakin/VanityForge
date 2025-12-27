import subprocess
import json
import os
import glob
import sys
import base58
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

    grind_args = []
    if prefix:
        grind_args.extend(["--starts-with", f"{prefix}:1"])
    elif suffix:
        grind_args.extend(["--ends-with", f"{suffix}:1"])
    else:
        error_msg = "No pattern provided (prefix or suffix required)"
        logging.error(error_msg)
        print(json.dumps({"found": False, "error": error_msg}))
        sys.exit(1)

    if case_sensitive == 'false':
        grind_args.append("--ignore-case")

    # Use solana-keygen from /usr/local/bin/ (installed in Dockerfile)
    solana_keygen_cmd = "/usr/local/bin/solana-keygen"

    command = [solana_keygen_cmd, "grind"] + grind_args

    logging.info(f"Running command: {command}")

    try:
        # Run the grind
        process = subprocess.run(command, capture_output=True, text=True, check=True)

        # Log stdout/stderr for debugging
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
    except FileNotFoundError:
        logging.error("solana-keygen not found in PATH")
        print(json.dumps({"found": False, "error": "solana-keygen executable not found"}))
        sys.exit(1)
    except Exception as e:
        logging.exception("Unexpected error")
        print(json.dumps({"found": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
