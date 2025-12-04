import os
import json
import logging
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import pubsub_v1
from google.cloud import firestore

# Configuration
PROJECT_ID = 'vanityforge'
TOPIC_ID = 'vanity-grind-jobs'

app = Flask(__name__)

# CORS Fix: Enable CORS
CORS(app)

# Initialize Google Cloud Clients
# We wrap in try/except to avoid crashing locally if creds are missing,
# though in Cloud Run they will be present.
try:
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
    db = firestore.Client(project=PROJECT_ID)
except Exception as e:
    logging.error(f"Failed to initialize Google Cloud clients: {e}")
    publisher = None
    topic_path = None
    db = None

@app.route('/', methods=['GET'])
def health_check():
    return "Dispatcher Service Operational", 200

@app.route('/submit-job', methods=['POST'])
def submit_job():
    # JSON Crash Fix: handle empty or malformed JSON
    data = request.get_json(silent=True) or {}

    if not data:
        return jsonify({'error': 'Invalid or empty JSON body'}), 400

    user_id = data.get('user_id')
    prefix = data.get('prefix')

    if not user_id or not prefix:
        return jsonify({'error': 'Missing required fields: user_id, prefix'}), 400

    # Create a unique job ID
    timestamp = int(datetime.datetime.now().timestamp())
    job_id = f"job_{user_id}_{timestamp}"

    # Prepare data for Firestore
    firestore_data = {
        'job_id': job_id,
        'user_id': user_id,
        'prefix': prefix,
        'status': 'queued',
        'created_at': firestore.SERVER_TIMESTAMP
    }

    # Prepare data for Pub/Sub (needs JSON serializable fields)
    pubsub_data = {
        'job_id': job_id,
        'user_id': user_id,
        'prefix': prefix,
        'created_at': datetime.datetime.now().isoformat()
    }

    try:
        # Save to Firestore
        if db:
            db.collection('vanity_jobs').document(job_id).set(firestore_data)

        # Publish to Pub/Sub
        if publisher and topic_path:
            data_str = json.dumps(pubsub_data)
            data_bytes = data_str.encode('utf-8')
            publish_future = publisher.publish(topic_path, data=data_bytes)
            # Wait for the publish to complete
            publish_future.result()

        return jsonify({'message': 'Job submitted successfully', 'job_id': job_id}), 200

    except Exception as e:
        logging.error(f"Error processing job: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
