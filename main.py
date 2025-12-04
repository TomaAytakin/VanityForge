import os
import json
import logging
import datetime
import uuid
from flask import Flask, request, jsonify, make_response
from google.cloud import pubsub_v1
from google.cloud import firestore

# Configuration
PROJECT_ID = 'vanityforge'
TOPIC_ID = 'vanity-grind-jobs'

app = Flask(__name__)

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

@app.after_request
def apply_cors(response):
    # This header ensures the browser accepts the final POST response
    response.headers['Access-Control-Allow-Origin'] = 'https://tomaaytakin.github.io'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/submit-job', methods=['POST', 'OPTIONS'])
def submit_job():
    # 1. CORS Preflight Handler
    if request.method == 'OPTIONS':
        # Return 204 No Content. Headers are handled by @app.after_request.
        return ('', 204)

    # 2. JSON Crash Fix: handle empty or malformed JSON
    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify({'error': 'Invalid or empty JSON body'}), 400
    # --- Application Logic ---
    user_id = data.get('userId')
    prefix = data.get('prefix')
    suffix = data.get('suffix', '')
    if not user_id or (not prefix and not suffix):
        return jsonify({'error': 'Missing required fields: user ID and either prefix or suffix.'}), 400
    # Generate simple unique ID (using uuid, assuming it's imported globally)
    job_id = str(uuid.uuid4())
    firestore_data = {
        'job_id': job_id,
        'user_id': user_id,
        'prefix': prefix,
        'suffix': suffix,
        'status': 'QUEUED',
        'cost': data.get('cost', 0),
        'created_at': firestore.SERVER_TIMESTAMP
    }
    pubsub_data = {
        'job_id': job_id,
        'prefix': prefix,
        'suffix': suffix,
        'user_id': user_id
    }
    try:
        # Save Job Status to Firestore
        db.collection('vanity_jobs').document(job_id).set(firestore_data)
        # Publish message to Pub/Sub
        data_str = json.dumps(pubsub_data)
        data_bytes = data_str.encode('utf-8')
        publisher.publish(topic_path, data=data_bytes).result()
        return jsonify({'message': 'Job submitted successfully', 'job_id': job_id}), 202
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {e}")
        return jsonify({'error': 'Internal Server Error during Queue submission.'}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
