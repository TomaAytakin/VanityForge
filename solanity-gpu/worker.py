import subprocess
import json
import os
import glob
import sys
import base58

def main():
    # Cloud Run Jobs pass arguments as indexed strings in sys.argv
    # The first arg is the script name, the rest are the payload
    raw_args = sys.argv[1:]
    
    prefix = ""
    suffix = ""
    
    # Simple parser for the way Cloud Run passes strings
    for i, arg in enumerate(raw_args):
        if arg == "--prefix" and i + 1 < len(raw_args):
            prefix = raw_args[i+1]
        if arg == "--suffix" and i + 1 < len(raw_args):
            suffix = raw_args[i+1]

    grind_args = []
    if prefix:
        grind_args.extend(["--starts-with", f"{prefix}:1"])
    elif suffix:
        grind_args.extend(["--ends-with", f"{suffix}:1"])
    else:
        print(json.dumps({"found": False, "error": "No pattern provided"}))
        sys.exit(1)

    command = ["solana-keygen", "grind"] + grind_args

    try:
        # Run the grind
        process = subprocess.run(command, capture_output=True, text=True, check=True)

        # Find the .json file solana-keygen just made
        json_files = glob.glob("*.json")
        if not json_files:
            print(json.dumps({"found": False, "error": "Keypair file not generated"}))
            sys.exit(1)

        found_file = json_files[0]
        with open(found_file, "r") as f:
            keypair_data = json.load(f)

        private_key_bytes = bytes(keypair_data)
        private_key_b58 = base58.b58encode(private_key_bytes).decode('utf-8')
        
        # Last 32 bytes are always the public key in a 64-byte Solana keypair
        public_key_bytes = private_key_bytes[32:]
        public_key_b58 = base58.b58encode(public_key_bytes).decode('utf-8')

        print(json.dumps({
            "found": True,
            "public_key": public_key_b58,
            "private_key": private_key_b58
        }))

        os.remove(found_file)

    except Exception as e:
        print(json.dumps({"found": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
