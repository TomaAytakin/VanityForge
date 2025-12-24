import os, sys, subprocess
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {'projectId': 'vanityforge'})
db = firestore.client()
def grind():
    job_id = os.environ.get('TASK_JOB_ID')
    prefix = os.environ.get('TASK_PREFIX', '')
    suffix = os.environ.get('TASK_SUFFIX', '')
    case = os.environ.get('TASK_CASE', 'false')

    print(f"⛏️ Rust CPU Miner starting Job {job_id}")
    db.collection('vanity_jobs').document(job_id).update({'status': 'GRINDING'})

    cmd = ["./cpu-grinder-bin", "--prefix", prefix, "--suffix", suffix, "--case-sensitive", case]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if "MATCH_FOUND:" in result.stdout:
            pkey = result.stdout.split("MATCH_FOUND:")[1].strip()
            db.collection('vanity_jobs').document(job_id).update({
                'status': 'SUCCESS', 'private_key': pkey, 'completed_at': firestore.SERVER_TIMESTAMP
            })
            print("✅ Job Complete!")
        else:
            raise Exception("No match in stdout")
    except Exception as e:
        print(f"❌ Error: {e}")
        db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': str(e)})

if __name__ == "__main__":
    grind()
