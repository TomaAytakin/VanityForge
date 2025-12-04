from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import pubsub_v1, firestore
import os
import uuid
import json
from datetime import datetime

# Configuration
# Verified: GCP_PROJECT_ID and PUBSUB_TOPIC_ID match Cloud Run settings.
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "vanityforge")
TOPIC_ID = os.environ.get("PUBSUB_TOPIC_ID", "vanity-grind-jobs")
JOB_COLLECTION = 'vanity_jobs'

app = Flask(__name__)
CORS(app)

# Initialize clients
try:
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
    db = firestore.Client()
except Exception as e:
    print(f"Warning: Clients failed to initialize (Local dev?): {e}")

def validate_input(data):
    if not data:
        return False, "No data provided"
    
    # Check if at least one constraint exists
    prefix = data.get('prefix', '')
    suffix = data.get('suffix', '')
    
    if not prefix and not suffix:
        return False, "Prefix or Suffix must be provided."
        
    # Base58 validation
    invalid_chars = ['0', 'O', 'I', 'l']
    for char in prefix + suffix:
        if char in invalid_chars or not char.isalnum():
            return False, f"Invalid Base58 character detected: '{char}'"
            
    return True, None

@app.route('/submit-job', methods=['POST'])
def submit_job():
    try:
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "Invalid or empty JSON body"}), 400

        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        job_id = str(uuid.uuid4())
        
        # 1. Save Initial Job Status to Firestore
        job_ref = db.collection(JOB_COLLECTION).document(job_id)
        job_ref.set({
            'job_id': job_id,
            'user_id': data.get('userId', 'anonymous'),
            'prefix': data.get('prefix', ''),
            'suffix': data.get('suffix', ''),
            'status': 'QUEUED',
            'cost': data.get('cost', 0),
            'created_at': firestore.SERVER_TIMESTAMP,
            'completed_at': None,
            'public_key': None
        })
        
        # 2. Publish message to Pub/Sub
        message_payload = json.dumps({
            'job_id': job_id, 
            'prefix': data.get('prefix', ''), 
            'suffix': data.get('suffix', '')
        })
        
        future = publisher.publish(topic_path, message_payload.encode('utf-8'))
        future.result() # Wait for publish
        
        return jsonify({"jobId": job_id, "status": "QUEUED"}), 202

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
