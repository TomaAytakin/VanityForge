#!/usr/bin/env python3
import sys
import json
import multiprocessing
import queue
import base58
from google.cloud import firestore
from solders.keypair import Keypair

def worker_task(prefix, suffix, stop_event, result_queue):
    """
    Worker function to generate keypairs until a match is found
    or the stop_event is set.
    """
    # Try to minimize overhead in the loop
    while not stop_event.is_set():
        kp = Keypair()
        pubkey = str(kp.pubkey())

        if pubkey.startswith(prefix) and pubkey.endswith(suffix):
            # We found a match!
            stop_event.set()
            # Convert bytes to list for JSON serialization
            secret_array = list(bytes(kp))
            result_queue.put((pubkey, secret_array))
            return

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

    # Initialize multiprocessing
    # We use all available CPUs
    try:
        num_workers = multiprocessing.cpu_count()
    except NotImplementedError:
        num_workers = 1

    stop_event = multiprocessing.Event()
    result_queue = multiprocessing.Queue()

    workers = []

    # Start workers
    for _ in range(num_workers):
        p = multiprocessing.Process(
            target=worker_task,
            args=(prefix, suffix, stop_event, result_queue)
        )
        p.start()
        workers.append(p)

    # Wait for a result
    try:
        # We block until one worker puts a result in the queue.
        # In a real scenario, we might want to handle signals to graceful shutdown.
        pubkey, secret_array = result_queue.get()
    except KeyboardInterrupt:
        stop_event.set()
        for p in workers:
            p.join()
        sys.exit(1)

    # Ensure all workers are stopped
    stop_event.set()
    for p in workers:
        p.join()

    # Initialize google.cloud.firestore (only in main process)
    try:
        secret_key_b58 = base58.b58encode(bytes(secret_array)).decode('utf-8')
        db = firestore.Client()
        doc_ref = db.collection("vanity_jobs").document(job_id)
        # Update Firestore with the result, including the secret key
        doc_ref.update({
            "status": "COMPLETED",
            "public_key": pubkey,
            "secret_key": secret_key_b58
        })
    except Exception as e:
        print(f"Error updating Firestore: {e}", file=sys.stderr)
        # We still print the key?
        # The requirements say "Update Firestore... Print the secret key".
        # Even if firestore fails, printing the key might be useful,
        # but usually if firestore fails we might want to log it.
        # I'll proceed to print.

    # Print secret key array to stdout
    print(json.dumps(secret_array))

if __name__ == "__main__":
    main()
