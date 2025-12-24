import os, sys, subprocess, hashlib, base64
import firebase_admin
import base58
from firebase_admin import credentials, firestore
from cryptography.fernet import Fernet

if not firebase_admin._apps:
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {'projectId': 'vanityforge'})
db = firestore.client()

def get_encryption_key(pin):
    # Hash PIN to create a 32-byte key for AES encryption
    digest = hashlib.sha256(str(pin).encode()).digest()
    return base64.urlsafe_b64encode(digest)

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
    # This PIN comes from vm_server.py when it dispatches the job
    user_pin = os.environ.get('TASK_PIN', '0000') 

    print(f"⛏️ Secure Miner starting Job {job_id}")
    db.collection('vanity_jobs').document(job_id).update({'status': 'GRINDING'})

    cmd = ["./cpu-grinder-bin", "--prefix", prefix, "--suffix", suffix, "--case-sensitive", case_sensitive]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if "MATCH_FOUND:" in result.stdout:
            raw_full_key = result.stdout.split("MATCH_FOUND:")[1].strip()
            
            # 1. Derive Public Key (Address) for display
            public_key = extract_public_key(raw_full_key)

            # 2. Encrypt the Private Key
            cipher = Fernet(get_encryption_key(user_pin))
            encrypted_private_key = cipher.encrypt(raw_full_key.encode()).decode()

            # 3. Save EVERYTHING to Firestore
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
        db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': str(e)})

if __name__ == "__main__":
    grind()

def get_encryption_key(pin):
    digest = hashlib.sha256(str(pin).encode()).digest()
    return base64.urlsafe_b64encode(digest)

def extract_public_key(full_key_b58):
    key_bytes = base58.b58decode(full_key_b58)
    public_key_bytes = key_bytes[32:]
    return base58.b58encode(public_key_bytes).decode('utf-8')

def grind():
    job_id = os.environ.get('TASK_JOB_ID')
    prefix = os.environ.get('TASK_PREFIX', '')
    suffix = os.environ.get('TASK_SUFFIX', '')
    case = os.environ.get('TASK_CASE', 'false')
    user_pin = os.environ.get('TASK_PIN', '0000') 

    print(f"⛏️ Secure Miner starting Job {job_id}")
    db.collection('vanity_jobs').document(job_id).update({'status': 'GRINDING'})

    cmd = ["./cpu-grinder-bin", "--prefix", prefix, "--suffix", suffix, "--case-sensitive", case]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if "MATCH_FOUND:" in result.stdout:
            raw_full = result.stdout.split("MATCH_FOUND:")[1].strip()
            
            pub_key = extract_public_key(raw_full)
            cipher = Fernet(get_encryption_key(user_pin))
            enc_priv = cipher.encrypt(raw_full.encode()).decode()

            db.collection('vanity_jobs').document(job_id).update({
                'status': 'COMPLETED', 'private_key': enc_priv, 
                'public_key': pub_key, 'is_encrypted': True,
                'completed_at': firestore.SERVER_TIMESTAMP
            })
            print(f"✅ Success! Address: {pub_key}")
        else:
            raise Exception("No match found")
    except Exception as e:
        print(f"❌ Error: {e}")
        db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': str(e)})

if __name__ == "__main__":
    grind()
