import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import pubsub_v1
from google.cloud import firestore

app = Flask(__name__)
CORS(app)

# Configuration (Hardcoded)
PROJECT_ID = 'vanityforge'
TOPIC_ID = 'vanity-grind-jobs'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
try:
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
    db = firestore.Client(project=PROJECT_ID)
except Exception as e:
    logger.error(f"Error initializing clients: {e}")
    publisher = None
    db = None

def validate_input(data):
    if not isinstance(data, dict):
        return False
    return True

@app.route('/submit-job', methods=['POST'])
def submit_job():
    data = request.get_json(silent=True) or {}

    if not data or not validate_input(data):
        return jsonify({"error": "Invalid or empty JSON body"}), 400

    if not publisher or not db:
        return jsonify({"error": "Service misconfigured (clients not initialized)"}), 500

    try:
        # Save job to Firestore
        doc_ref = db.collection('vanity_jobs').document()
        job_id = doc_ref.id

        job_data = data.copy()
        job_data['status'] = 'PENDING'
        job_data['created_at'] = firestore.SERVER_TIMESTAMP
        job_data['job_id'] = job_id

        doc_ref.set(job_data)
        logger.info(f"Job saved to Firestore with ID: {job_id}")

        # Publish to Pub/Sub
        # Prepare message data (exclude non-serializable fields like SERVER_TIMESTAMP)
        message_data = job_data.copy()
        if 'created_at' in message_data:
            del message_data['created_at']

        message_json = json.dumps(message_data).encode("utf-8")
        future = publisher.publish(topic_path, message_json)
        message_id = future.result()
        logger.info(f"Job published to Pub/Sub with Message ID: {message_id}")

        return jsonify({
            "message": "Job submitted successfully",
            "job_id": job_id,
            "message_id": message_id
        }), 200

    except Exception as e:
        logger.error(f"Error processing job: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
