import os, sys, subprocess, hashlib, base64
import firebase_admin
import base58
from firebase_admin import credentials, firestore
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Check if Firebase is already initialized to avoid "ValueError: The default Firebase app already exists"
if not firebase_admin._apps:
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {'projectId': 'vanityforge'})
db = firestore.client()

def get_deterministic_salt(user_id):
    return hashlib.sha256(user_id.encode()).digest()

def generate_key_from_pin(pin, user_id):
    # Match server logic: PBKDF2 with deterministic salt from user_id
    salt = get_deterministic_salt(user_id)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))

def extract_public_key(full_key_b58):
    # The miner returns a 64-byte keypair in Base58
    # Bytes 0-31 = Private Seed
    # Bytes 32-63 = Public Key (Address)
    key_bytes = base58.b58decode(full_key_b58)
    public_key_bytes = key_bytes[32:]
    return base58.b58encode(public_key_bytes).decode('utf-8')

def grind():
    job_id = os.environ.get('TASK_JOB_ID')
    prefix = os.environ.get('TASK_PREFIX', '')
    suffix = os.environ.get('TASK_SUFFIX', '')
    case_sensitive = os.environ.get('TASK_CASE', 'false')
    user_pin = os.environ.get('TASK_PIN', '0000') 
    user_id = os.environ.get('TASK_USER_ID', '')

    if not user_id:
        print("❌ Error: TASK_USER_ID is missing. Cannot derive encryption key securely.")
        if job_id:
            db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': 'Missing TASK_USER_ID'})
        sys.exit(1)

    print(f"⛏️ Secure Miner starting Job {job_id}")
    if job_id:
        db.collection('vanity_jobs').document(job_id).update({'status': 'GRINDING'})

    cmd = ["./cpu-grinder-bin", "--prefix", prefix, "--suffix", suffix, "--case-sensitive", case_sensitive]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if "MATCH_FOUND:" in result.stdout:
            raw_full_key = result.stdout.split("MATCH_FOUND:")[1].strip()
            
            # 1. Derive Public Key (Address) for display
            public_key = extract_public_key(raw_full_key)

            # 2. Encrypt the Private Key
            # Use generate_key_from_pin with user_id to match server logic
            enc_key = generate_key_from_pin(user_pin, user_id)
            cipher = Fernet(enc_key)
            encrypted_private_key = cipher.encrypt(raw_full_key.encode()).decode()

            # 3. Save EVERYTHING to Firestore
            if job_id:
                db.collection('vanity_jobs').document(job_id).update({
                    'status': 'COMPLETED',
                    'private_key': encrypted_private_key, # Locked
                    'public_key': public_key,             # Visible (Fixes "undefined")
                    'is_encrypted': True,
                    'completed_at': firestore.SERVER_TIMESTAMP
                })
            print(f"✅ Success! Address: {public_key}")
        else:
            raise Exception("No match found")
    except Exception as e:
        print(f"❌ Error: {e}")
        if job_id:
            db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': str(e)})

if __name__ == "__main__":
    grind()
