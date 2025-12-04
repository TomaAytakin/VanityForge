import os
import json
import logging
import multiprocessing
import time
from google.cloud import pubsub_v1
from google.cloud import firestore
import nacl.signing
import base58

# Hardcoded configuration
PROJECT_ID = 'vanityforge'
TOPIC_ID = 'vanity-grind-jobs'
SUBSCRIPTION_ID = 'vanity-grind-jobs-sub'

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_keypair():
    """Generates a Solana keypair."""
    signing_key = nacl.signing.SigningKey.generate()
    verify_key = signing_key.verify_key

    # Solana secret key is the 64-byte concatenation of the seed and the public key
    secret_key_bytes = signing_key.encode() + verify_key.encode()
    public_key_bytes = verify_key.encode()

    return base58.b58encode(public_key_bytes).decode('ascii'), base58.b58encode(secret_key_bytes).decode('ascii')

def grind_worker(prefix, suffix, case_sensitive, result_queue, stop_event):
    """Worker process to grind vanity addresses."""
    # Normalize targets for faster comparison loop if case insensitive
    target_prefix = prefix
    target_suffix = suffix
    if not case_sensitive:
        if target_prefix: target_prefix = target_prefix.lower()
        if target_suffix: target_suffix = target_suffix.lower()

    while not stop_event.is_set():
        pub_key, priv_key = generate_keypair()

        check_pub = pub_key
        if not case_sensitive:
            check_pub = check_pub.lower()

        if target_prefix and not check_pub.startswith(target_prefix):
            continue
        if target_suffix and not check_pub.endswith(target_suffix):
            continue

        # Found!
        result_queue.put((pub_key, priv_key))
        return

def process_job(job_data, job_id):
    """Processes a single job using multiprocessing."""
    prefix = job_data.get('prefix', '')
    suffix = job_data.get('suffix', '')
    case_sensitive = job_data.get('case_sensitive', True)

    logging.info(f"Starting grind for Job {job_id}: Prefix='{prefix}', Suffix='{suffix}', CaseSensitive={case_sensitive}")

    num_cores = multiprocessing.cpu_count()
    logging.info(f"Using {num_cores} cores.")

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    stop_event = manager.Event()

    processes = []
    for _ in range(num_cores):
        p = multiprocessing.Process(target=grind_worker, args=(prefix, suffix, case_sensitive, result_queue, stop_event))
        p.start()
        processes.append(p)

    # Wait for result
    pub_key, priv_key = result_queue.get() # This blocks until one is found

    # Stop all workers
    stop_event.set()
    for p in processes:
        p.terminate()
        p.join()

    logging.info(f"Job {job_id} COMPLETED. Found: {pub_key}")
    return pub_key, priv_key

def callback(message):
    try:
        data = json.loads(message.data.decode('utf-8'))
        job_id = data.get('job_id')

        if not job_id:
            logging.error("Received message without job_id")
            message.ack()
            return

        logging.info(f"Received job: {job_id}")

        # Extend ack deadline or keep alive is not implemented here for simplicity
        # Assuming finding the key happens within reasonable time or redelivery handles it.

        pub_key, secret_key = process_job(data, job_id)

        # Update Firestore
        db = firestore.Client(project=PROJECT_ID)
        doc_ref = db.collection('vanity_jobs').document(job_id)
        doc_ref.update({
            'public_key': pub_key,
            'secret_key': secret_key,
            'status': 'COMPLETED'
        })

        logging.info(f"Updated Firestore for job {job_id}")
        message.ack()

    except Exception as e:
        logging.error(f"Error processing message: {e}")
        message.nack()

def main():
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

    # Attempt to create subscription if it doesn't exist (optional helper)
    try:
        topic_path = subscriber.topic_path(PROJECT_ID, TOPIC_ID)
        with pubsub_v1.PublisherClient() as publisher:
             # Just checking if we can access the project/topic logic
             # usually creation is done via Terraform or CLI, but this helps standalone run
             pass
        # subscriber.create_subscription(name=subscription_path, topic=topic_path)
    except Exception:
        pass

    logging.info(f"Listening for messages on {subscription_path}...")

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)

    with subscriber:
        try:
            streaming_pull_future.result()
        except TimeoutError:
            streaming_pull_future.cancel()

if __name__ == '__main__':
    main()
