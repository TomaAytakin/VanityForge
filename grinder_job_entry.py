import os
import json
import time
import uuid
import base58
import multiprocessing
import logging
from google.cloud import firestore
from solders.keypair import Keypair

PROJECT_ID = os.environ.get('PROJECT_ID', 'vanityforge')

def grind_key(prefix, event, result_queue):
    try:
        while not event.is_set():
            kp = Keypair()
            pubkey = str(kp.pubkey())

            if pubkey.startswith(prefix):
                secret_bytes = bytes(kp)
                secret_b58 = base58.b58encode(secret_bytes).decode('ascii')
                result_queue.put({
                    'public_key': pubkey,
                    'secret_key': secret_b58
                })
                event.set()
                return
    except Exception:
        logging.exception("Worker error")
        return

def process_job(job_id, prefix):
    if not job_id or not prefix:
        logging.error("Missing job_id or prefix")
        return None

    logging.info(f"Grinding for job {job_id} prefix '{prefix}' using {multiprocessing.cpu_count()} cores")

    manager = multiprocessing.Manager()
    event = manager.Event()
    result_queue = manager.Queue()

    workers = []
    for _ in range(max(1, multiprocessing.cpu_count())):
        p = multiprocessing.Process(target=grind_key, args=(prefix, event, result_queue))
        p.start()
        workers.append(p)

    event.wait()

    result = None
    try:
        result = result_queue.get_nowait()
    except Exception:
        logging.exception("Result missing")

    for p in workers:
        try:
            p.terminate()
            p.join(timeout=5)
        except Exception:
            pass

    logging.info(f"Job {job_id} completed: {result}")
    return result

def update_firestore_on_start(job_id):
    try:
        db = firestore.Client(project=PROJECT_ID)
        db.collection('vanity_jobs').document(job_id).update({
            'status': 'RUNNING',
            'started_at': firestore.SERVER_TIMESTAMP
        })
    except Exception:
        logging.exception("Failed to mark job running")

def update_firestore_on_complete(job_id, result):
    try:
        db = firestore.Client(project=PROJECT_ID)
        db.collection('vanity_jobs').document(job_id).update({
            'status': 'COMPLETED' if result else 'FAILED',
            'public_key': result.get('public_key') if result else None,
            'secret_key': result.get('secret_key') if result else None,
            'completed_at': firestore.SERVER_TIMESTAMP
        })
    except Exception:
        logging.exception("Failed to update Firestore")

def main():
    logging.basicConfig(level=logging.INFO)

    job_id = os.environ.get('JOB_ID')
    prefix = os.environ.get('PREFIX')

    logging.info(f"JOB START | JOB_ID={job_id} PREFIX={prefix}")

    if not job_id or not prefix:
        logging.error("JOB_ID and PREFIX must be set for this job")
        return 2

    update_firestore_on_start(job_id)

    result = process_job(job_id, prefix)

    update_firestore_on_complete(job_id, result)

    return 0 if result else 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
