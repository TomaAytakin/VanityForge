import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import pubsub_v1
from google.cloud import firestore

# Hardcoded configuration
PROJECT_ID = 'vanityforge'
TOPIC_ID = 'vanity-grind-jobs'

app = Flask(__name__)
CORS(app)

# Initialize clients
try:
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
    db = firestore.Client(project=PROJECT_ID)
except Exception as e:
    logging.error(f"Failed to initialize Google Cloud clients: {e}")
    publisher = None
    db = None
    topic_path = None

@app.route('/submit-job', methods=['POST'])
def submit_job():
    data = request.get_json(silent=True) or {}

    if not data:
        return jsonify({"error": "Invalid or empty JSON body"}), 400

    try:
        if db:
            doc_ref = db.collection('vanity_jobs').document()
            job_id = doc_ref.id
        else:
            # Fallback for when DB is not available
            import uuid
            job_id = str(uuid.uuid4())
            doc_ref = None

        # Prepare data for Firestore
        job_data = data.copy()
        job_data['job_id'] = job_id
        job_data['status'] = 'PENDING'
        job_data['created_at'] = firestore.SERVER_TIMESTAMP

        # Save to Firestore
        if doc_ref:
            doc_ref.set(job_data)

        # Publish to Pub/Sub
        message_id = None
        if publisher and topic_path:
            # Prepare data for Pub/Sub (remove non-serializable fields if any)
            pubsub_payload = data.copy()
            pubsub_payload['job_id'] = job_id

            message_bytes = json.dumps(pubsub_payload).encode('utf-8')
            future = publisher.publish(topic_path, message_bytes)
            message_id = future.result()

        return jsonify({"job_id": job_id, "message_id": message_id}), 200

    except Exception as e:
        logging.exception("An error occurred during job submission")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
