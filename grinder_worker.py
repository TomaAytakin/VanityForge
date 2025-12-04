import sys
import json
from google.cloud import firestore
from solders.keypair import Keypair

def main():
    # Read from stdin
    try:
        input_data = sys.stdin.read()
        if not input_data:
            return

        message = json.loads(input_data)
    except json.JSONDecodeError:
        print("Invalid JSON input", file=sys.stderr)
        sys.exit(1)

    # Parse fields
    job_id = message.get("job_id")
    prefix = message.get("prefix", "")
    suffix = message.get("suffix", "")

    if not job_id:
        print("Missing job_id", file=sys.stderr)
        sys.exit(1)

    # Initialize google.cloud.firestore
    try:
        db = firestore.Client()
    except Exception as e:
        print(f"Error initializing Firestore: {e}", file=sys.stderr)
        sys.exit(1)

    # Loop generating keypairs
    while True:
        kp = Keypair()
        pubkey = str(kp.pubkey())

        if pubkey.startswith(prefix) and pubkey.endswith(suffix):
            # Match found

            # Update Firestore
            try:
                doc_ref = db.collection("vanity_jobs").document(job_id)
                doc_ref.update({
                    "status": "COMPLETED",
                    "public_key": pubkey
                })
            except Exception as e:
                print(f"Error updating Firestore: {e}", file=sys.stderr)

            # Print secret key array to stdout
            secret_array = list(bytes(kp))
            print(json.dumps(secret_array))
            break

if __name__ == "__main__":
    main()
