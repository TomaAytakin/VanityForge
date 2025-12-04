import os
import json
import time
import multiprocessing
import base58
from google.cloud import pubsub_v1
from google.cloud import firestore
from solders.keypair import Keypair

# Configuration
PROJECT_ID = 'vanityforge'
TOPIC_ID = 'vanity-grind-jobs'
SUBSCRIPTION_ID = 'vanity-grind-jobs-sub'

def grind_key(prefix, event, result_queue):
    """
    Worker function to run in a separate process.
    Continuously generates keys until one matching the prefix is found
    or the event is set (signaling another process found it).
    """
    try:
        while not event.is_set():
            kp = Keypair()
            pubkey = str(kp.pubkey())

            if pubkey.startswith(prefix):
                # Found a match!
                # Serialize the private key
                # solders Keypair bytes is the 64-byte secret (seed + pubkey)
                secret_bytes = bytes(kp)
                secret_b58 = base58.b58encode(secret_bytes).decode('ascii')

                result_queue.put({
                    'public_key': pubkey,
                    'secret_key': secret_b58
                })
                event.set()
                return
    except Exception as e:
        # In case of error, just return so process dies
        pass

def process_job(job_data):
    """
    Orchestrates the multicore grinding for a single job.
    """
    job_id = job_data.get('job_id')
    prefix = job_data.get('prefix')

    if not job_id or not prefix:
        print(f"Invalid job data: {job_data}")
        return None

    print(f"Starting grind for job {job_id} with prefix {prefix} on {multiprocessing.cpu_count()} cores.")

    manager = multiprocessing.Manager()
    event = manager.Event()
    result_queue = manager.Queue()

    workers = []
    # Spawn a process for each CPU core
    for _ in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(
            target=grind_key,
            args=(prefix, event, result_queue)
        )
        p.start()
        workers.append(p)

    # Wait for the event to be set (meaning a key was found)
    event.wait()

    # Retrieve result
    result = result_queue.get()

    # Terminate all workers
    for p in workers:
        p.terminate()
        p.join()

    print(f"Job {job_id} completed. Found address: {result['public_key']}")
    return result

def pubsub_callback(message):
    """
    Callback for Pub/Sub messages.
    """
    try:
        print(f"Received message: {message.data}")
        data = json.loads(message.data.decode('utf-8'))

        # Process the job using all cores
        result = process_job(data)

        if result:
            # Save to Firestore
            db = firestore.Client(project=PROJECT_ID)
            job_ref = db.collection('vanity_jobs').document(data['job_id'])

            job_ref.update({
                'status': 'COMPLETED',
                'secret_key': result['secret_key'], # Saved as unencrypted string as requested
                'public_key': result['public_key'],
                'completed_at': firestore.SERVER_TIMESTAMP
            })

            print(f"Updated Firestore for job {data['job_id']}")
            message.ack()
        else:
            message.ack()

    except Exception as e:
        print(f"Error processing message: {e}")
        message.nack()

def main():
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

    print(f"Listening for messages on {subscription_path}...")

    # Limit flow control to 1 message at a time to ensure we dedicate all cores to the current job
    flow_control = pubsub_v1.types.FlowControl(max_messages=1)

    future = subscriber.subscribe(subscription_path, callback=pubsub_callback, flow_control=flow_control)

    try:
        future.result()
    except KeyboardInterrupt:
        future.cancel()

if __name__ == '__main__':
    main()
