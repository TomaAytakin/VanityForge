import os
import uuid
import logging
import threading
import base58
import re
import time
import signal
import base64
import bcrypt
import hashlib
import math
import random
import string
import requests
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, jsonify, render_template, abort, session
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
from google.cloud import firestore
from google.cloud import run_v2
import firebase_admin
from firebase_admin import credentials, auth
from solders.keypair import Keypair
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
from utils.stats_engine import calculate_marketing_stats

# 1. LOAD SECRETS
load_dotenv()

# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 2. CONFIGURATION
PROJECT_ID = os.getenv('PROJECT_ID', 'vanityforge').strip()
SOLANA_RPC_URL = os.getenv('SOLANA_RPC_URL', '').strip()
TREASURY_PUBKEY = os.getenv('TREASURY_PUBKEY', '').strip()
SMTP_EMAIL = os.getenv('SMTP_EMAIL', '').strip()
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '').strip()
os.environ.setdefault("SMTP_SERVER", "smtp.gmail.com")
os.environ.setdefault("SMTP_PORT", "587")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')

# --- ADMIN LIST (GOD MODE) ---
ADMIN_EMAILS = set(os.getenv('ADMIN_EMAILS', "tomaaytakin@gmail.com,admin@vanityforge.org,Jonny95hidalgo@gmail.com").split(','))

# --- QUEUE LIMITS (TRAFFIC CONTROL) ---
MAX_CLOUD_JOBS = 100  # Max concurrent jobs on Cloud Run

# CLOUD JOB CONFIGURATION
REDPANDA_JOB_NAME = os.getenv('REDPANDA_JOB_NAME', "projects/vanityforge/locations/us-central1/jobs/cpu-redpanda")

# 3. SAFETY CHECK
if not SOLANA_RPC_URL or not TREASURY_PUBKEY or not SMTP_PASSWORD:
    logging.critical("CRITICAL ERROR: Missing environment variables. Make sure .env exists.")
    exit(1)

# --- FIREBASE ADMIN INIT ---
try:
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    logging.info("🔥 Firebase Admin Initialized")
except Exception as e:
    logging.warning(f"⚠️ Firebase Admin Init Failed: {e}")

app = Flask(__name__, static_folder='.', static_url_path='', template_folder='.')
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24).hex())

# Fix IP logging to trust Google Cloud Load Balancer headers
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

@app.before_request
def block_hidden_paths():
    # Block any path starting with '.' or containing '/.' (e.g. /.env, /.git/config)
    if request.path.startswith("/.") or "/." in request.path:
        real_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        app.logger.warning(f"🚨 SECURITY: Blocked attempt to access {request.path} from {real_ip}")
        abort(403)

@app.after_request
def set_security_and_caching_headers(response):
    # --- SECURITY HEADERS ---
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Content-Security-Policy'] = "default-src 'self' https: data: 'unsafe-inline' 'unsafe-eval';"

    # --- CACHING HEADERS (INFRASTRUCTURE SUPPORT) ---
    path = request.path

    # List of dynamic API endpoints that must NEVER be cached
    # Note: /api/* covers most, but specific routes are listed for safety
    no_cache_paths = [
        '/submit-job', '/check-user', '/set-pin', '/reveal-key'
    ]

    is_api = path.startswith('/api/') or path in no_cache_paths

    if is_api:
        # Dynamic: Bypass CDN & Browser Cache
        response.headers['Cache-Control'] = 'private, no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    else:
        # Static: Enable Cloud CDN Caching
        # Applies to: /, /roadmap, /faq, /vvvip, *.html, *.css, *.js, *.json, *.png, etc.
        # We use public caching with a 1-hour max-age
        response.headers['Cache-Control'] = 'public, max-age=3600'

    return response

CORS(app, resources={r"/*": {"origins": ["https://tomaaytakin.github.io", "https://vanityforge.org"]}})

# --- RATE LIMITER CONFIGURATION ---
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://"
)

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded'}), 429

def is_admin_session():
    """Checks if the current session belongs to an admin."""
    email = session.get('email')
    if not email: return False

    # 1. Existing/Global Config (Plural)
    admins = globals().get('ADMIN_EMAILS') or set(os.getenv('ADMIN_EMAILS', '').split(','))

    # 2. User Requested Config (Singular) - "Strictly rely on ADMIN_EMAIL"
    single_admin = os.getenv('ADMIN_EMAIL')
    if single_admin:
        admins.add(single_admin)

    return email in admins

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_admin_session():
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index(): return app.send_static_file('index.html')

@app.route('/roadmap')
def roadmap(): return app.send_static_file('roadmap.html')

@app.route('/vvvip')
def vvvip():
    rpc_url = os.getenv("SOLANA_RPC_URL", "https://rpc.ankr.com/solana")
    return render_template('vvvip.html', rpc_url=rpc_url)

@app.route('/faq')
def faq(): return app.send_static_file('faq.html')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Returns the server-side calculated forged count."""
    count = calculate_marketing_stats()
    return jsonify({'forged_count': count})

# --- RPC PROXY ---
@app.route('/api/rpc', methods=['POST'])
@limiter.limit("100 per minute")
def rpc_proxy():
    data = request.get_json(silent=True) or {}
    try:
        resp = requests.post(SOLANA_RPC_URL, json=data, timeout=10)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- ADMIN / GOD MODE ENDPOINTS ---
@app.route('/api/user-status', methods=['GET'])
def user_status():
    """Returns the admin status and email of the current session."""
    return jsonify({
        'isAdmin': is_admin_session(),
        'email': session.get('email')
    })

@app.route('/api/admin/stats', methods=['GET'])
@login_required
def admin_stats():
    return jsonify({
        'income_1w': 12.5,
        'income_1m': 45.0,
        'income_1y': 150.2,
        'total_users': 1337,
        'server_health': 'OK'
    })

@app.route('/api/admin/security', methods=['GET'])
@login_required
def admin_security():
    return jsonify([
        {'ip': '1.2.3.4', 'reason': 'Rate Limit Exceeded', 'time': '10 mins ago'},
        {'ip': '5.6.7.8', 'reason': 'Path Traversal', 'time': '1 hour ago'}
    ])

@app.route('/api/admin/referrals', methods=['GET'])
@login_required
def admin_referrals():
    return jsonify([
        {'code': 'ABCD', 'earnings': 10.5},
        {'code': 'XYZ1', 'earnings': 5.2}
    ])

# --- REFERRAL SYSTEM HELPERS ---
def generate_referral_code():
    """Generates a unique 4-character alphanumeric code."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

def cleanup_firestore_client(db):
    """Safely closes the Firestore client connection."""
    try:
        if db:
            db.close()
    except Exception:
        # Fail silently during cleanup to keep logs clean
        pass

def is_base58(s):
    if not s: return True
    return bool(re.match(r'^[1-9A-HJ-NP-Za-km-z]+$', s))

def calculate_top64_range(prefix_str):
    """Calculates the Big-Endian uint64 range for the GPU Rejector."""
    if not prefix_str:
        return 0, 0xFFFFFFFFFFFFFFFF

    try:
        p_bytes = base58.b58decode(prefix_str)
        P = int.from_bytes(p_bytes, byteorder='big')
    except ValueError:
        return 0, 0

    # Target Length ~44 chars
    shift_power = 44 - len(prefix_str)
    if shift_power < 0: shift_power = 0
    shift_factor = 58 ** shift_power

    min_int = P * shift_factor
    max_int = (P + 1) * shift_factor - 1
    max_256 = 2**256 - 1

    if min_int > max_256: return 0, 0 # Impossible
    if max_int > max_256: max_int = max_256

    def get_top_64(num):
        b = num.to_bytes(32, byteorder='big')
        return int.from_bytes(b[:8], byteorder='big')

    min_limit = get_top_64(min_int)
    max_limit = get_top_64(max_int)

    # Safety Widening
    min_limit = max(0, min_limit - 4)
    max_limit = min(0xFFFFFFFFFFFFFFFF, max_limit + 4)

    return min_limit, max_limit

# --- DISPATCHER LOGIC (CLOUD RUN JOBS) ---
def dispatch_cloud_job(job_id, user_id, prefix, suffix, case_sensitive, pin):
    """Triggers the remote Cloud Run Job (Titanium GPU Cluster)."""
    try:
        target_job = REDPANDA_JOB_NAME
        logging.info("🏎️ Dispatching to Titanium GPU Cluster")

        min_limit, max_limit = calculate_top64_range(prefix)

        # Build Arguments List (Correct Format: ["--prefix", "val", "--suffix", "val"])
        args = []
        if prefix:
            args.extend(["--prefix", prefix])
        if suffix:
            args.extend(["--suffix", suffix])

        # CPU Grinder expects string "true" for bool flag based on logic
        if case_sensitive:
            args.extend(["--case-sensitive", "true"])

        client = run_v2.JobsClient()
        request = run_v2.RunJobRequest(
            name=target_job,
            overrides={
                "container_overrides": [{
                    "args": args,
                    "env": [
                        {"name": "TASK_JOB_ID", "value": job_id},
                        {"name": "TASK_PREFIX", "value": prefix or ""},
                        {"name": "TASK_SUFFIX", "value": suffix or ""},
                        {"name": "TASK_CASE", "value": str(case_sensitive)},
                        {"name": "TASK_PIN", "value": str(pin)},
                        {"name": "TASK_USER_ID", "value": str(user_id)},
                        {"name": "TASK_MIN_LIMIT", "value": str(min_limit)},
                        {"name": "TASK_MAX_LIMIT", "value": str(max_limit)},
                        {"name": "SERVER_URL", "value": os.getenv("SERVER_URL", "https://vanityforge.org")}
                    ]
                }]
            }
        )
        operation = client.run_job(request=request)
        logging.info(f"🚀 Dispatched Cloud Job: {job_id}")
    except Exception as e:
        logging.exception(f"❌ Failed to dispatch Cloud job")
        # Mark as failed in DB so user isn't stuck forever
        db = firestore.Client(project=PROJECT_ID)
        try:
            db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': 'Cloud Dispatch Failed'})
        finally:
            cleanup_firestore_client(db)

# --- LOCAL CPU LOGIC ---
def check_user_trials(user_id):
    db = None
    try:
        db = firestore.Client(project=PROJECT_ID)
        docs = db.collection('vanity_jobs').where('user_id', '==', user_id).stream()
        return sum(1 for doc in docs if doc.to_dict().get('is_trial'))
    except: return 2
    finally:
        cleanup_firestore_client(db)

def verify_payment(signature, required_price_sol):
    if not signature: return False
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "getTransaction",
        "params": [signature, {"encoding": "json", "commitment": "confirmed", "maxSupportedTransactionVersion": 0}]
    }
    try:
        resp = requests.post(SOLANA_RPC_URL, json=payload, timeout=10)
        data = resp.json()
        if "error" in data or not data.get("result"): return False
        
        result = data.get("result")
        if result.get("meta", {}).get("err"): return False

        meta = result.get("meta", {})
        msg = result.get("transaction", {}).get("message", {})
        account_keys_raw = msg.get("accountKeys", [])
        accounts = [k.get("pubkey") if isinstance(k, dict) else k for k in account_keys_raw]

        try:
            idx = accounts.index(TREASURY_PUBKEY)
        except ValueError:
            logging.warning(f"Treasury not found in tx")
            return False

        received_sol = (meta.get("postBalances")[idx] - meta.get("preBalances")[idx]) / 1_000_000_000
        
        # Allow tiny variance for fees
        if received_sol < (required_price_sol - 0.0001):
            logging.warning(f"Insufficient: {received_sol} < {required_price_sol}")
            return False
        return True
    except Exception as e:
        logging.exception(f"Verify Error")
        return False

def get_deterministic_salt(user_id):
    return hashlib.sha256(user_id.encode()).digest()

def generate_key_from_pin(pin, user_id):
    salt = get_deterministic_salt(user_id)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))

# --- SOLA EMAIL ENGINE ---
SOLA_BANNER = "https://raw.githubusercontent.com/TomaAytakin/VanityForge/main/assets/Email%20Assets/vfemailbanner.png"
SOLA_PAW = "https://raw.githubusercontent.com/TomaAytakin/VanityForge/main/assets/Email%20Assets/vfemailsig.png"

def get_sola_html(body):
    """Wraps content in the Sola/Red Panda Branding"""
    return f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; color: #333;">
        <img src="{SOLA_BANNER}" width="100%" style="border-radius: 8px 8px 0 0;" alt="VanityForge Banner">
        <div style="padding: 20px; background-color: #ffffff; border: 1px solid #e0e0e0; border-top: none;">
            {body}
        </div>
        <div style="padding: 20px; background-color: #f9f9f9; border: 1px solid #e0e0e0; border-top: none; border-radius: 0 0 8px 8px;">
            <table style="width: 100%;">
                <tr>
                    <td style="width: 60px; vertical-align: middle;">
                        <img src="{SOLA_PAW}" width="50" height="50" alt="Paw">
                    </td>
                    <td style="vertical-align: middle;">
                        <strong style="color: #E55039; font-size: 16px;">Sola</strong> 🐼<br>
                        <span style="color: #666; font-size: 12px;">Chief Forging Scout @ VanityForge</span><br>
                        <a href="https://vanityforge.org" style="color: #6c5ce7; text-decoration: none; font-size: 12px;">vanityforge.org</a>
                    </td>
                </tr>
            </table>
        </div>
    </div>
    """

def send_email_wrapper(to_email, subject, html_content):
    """Handles the Alias Spoofing (Login as Admin, Send as Support)"""
    try:
        # Explicitly set the 'From' header to vanity address while logging in with SMTP creds
        sender_display = "VanityForge Support <support@vanityforge.org>"
        login_email = os.getenv('SMTP_EMAIL')
        login_password = os.getenv('SMTP_PASSWORD')

        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_display
        msg['To'] = to_email

        msg.attach(MIMEText(html_content, 'html'))

        context = ssl.create_default_context()
        with smtplib.SMTP(os.getenv('SMTP_SERVER'), int(os.getenv('SMTP_PORT'))) as server:
            server.starttls(context=context)
            server.login(login_email, login_password)
            # Send using login_email as envelope sender, but msg['From'] is displayed to user
            server.sendmail(login_email, to_email, msg.as_string())

        logging.info(f"📧 Sola Email sent to {to_email}")
        return True
    except Exception as e:
        logging.error(f"❌ Email Failed: {e}")
        return False

def send_start_email(to_email, job_id, prefix, suffix, price, is_trial):
    # Calculate Pricing Logic for Receipt
    total_len = len(prefix or '') + len(suffix or '')
    if total_len <= 4: base = 0.25
    elif total_len == 5: base = 0.50
    elif total_len == 6: base = 1.00
    elif total_len == 7: base = 2.00
    elif total_len == 8: base = 3.00
    else: base = 5.00

    # Standard Beta Price is 50% of Base
    standard_price = base * 0.5

    if is_trial:
        # Free Trial: Show strikethrough original price -> 0 SOL
        price_display = f"<s>{standard_price} SOL</s> 0 SOL (Free Trial)"
    else:
        # Paid Job: Show actual paid amount
        price_display = f"{price} SOL"

    receipt_html = f"""
    <h2 style="color: #2d3436;">Forge Started! ⚒️</h2>
    <p>Hey Degen! 🐾</p>
    <p>Sola here. I've successfully dispatched your job to the grinder.</p>
    <div style="background: #eee; padding: 10px; border-radius: 5px; margin: 15px 0;">
        <strong>Target:</strong> ...{suffix} (or {prefix}...)<br>
        <strong>Job ID:</strong> {job_id}<br>
        <strong>Price:</strong> {price_display}
    </div>
    <p>Sit tight! I'll ping you the second it's ready.</p>
    """
    send_email_wrapper(to_email, "Forge Started: Your Custom Wallet is Brewing! ⚒️", get_sola_html(receipt_html))

def send_completion_email(to_email, public_key, private_key_enc=None):
    success_html = f"""
    <h2 style="color: #00b894;">Forge Complete! 🚀</h2>
    <p>Great news from the bamboo forest! 🎋</p>
    <p>Your custom vanity wallet has been found.</p>
    <div style="border: 2px dashed #00b894; padding: 15px; text-align: center; margin: 20px 0;">
        <strong>Public Key:</strong><br>
        <code style="font-size: 14px; word-break: break-all;">{public_key}</code>
    </div>
    <p style="text-align: center;">
        <a href="https://vanityforge.org" style="background-color: #6c5ce7; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; font-weight: bold;">Reveal Private Key</a>
    </p>
    """
    send_email_wrapper(to_email, "Forge Complete! Your Wallet is Ready 🚀", get_sola_html(success_html))

# --- QUEUE SCHEDULER (THE TRAFFIC CONTROLLER) ---
def scheduler_loop():
    """Background thread that checks for open slots and starts queued jobs."""
    logging.info("✅ Queue Scheduler Started")
    db = firestore.Client(project=PROJECT_ID)
    while True:
        try:
            # 1. Count CURRENTLY running jobs
            running_cloud = 0
            running_docs = db.collection('vanity_jobs').where('status', '==', 'RUNNING').stream()
            for d in running_docs:
                running_cloud += 1
            
            # 2. Fetch QUEUED jobs (Oldest first = Fairness)
            # THIS QUERY REQUIRES THE COMPOSITE INDEX: status=QUEUED, order_by=created_at
            queued_docs = db.collection('vanity_jobs').where('status', '==', 'QUEUED').order_by('created_at').limit(10).stream()
            
            for doc in queued_docs:
                try:
                    data = doc.to_dict()
                    job_id = data.get('job_id')

                    # Retrieve the temporary PIN needed for encryption
                    pin_plain = data.get('temp_pin')
                    if not pin_plain:
                        # Error state: Lost PIN? Fail job to clear queue
                        logging.error(f"❌ Job {job_id} missing temp_pin, marking FAILED")
                        db.collection('vanity_jobs').document(job_id).update({'status': 'FAILED', 'error': 'PIN missing in queue'})
                        continue

                    # --- DISPATCH LOGIC ---
                    # RULE: ALL jobs now go to Cloud Run
                    if running_cloud < MAX_CLOUD_JOBS:
                        logging.info(f"🚦 Dispatching Cloud Job: {job_id}")
                        db.collection('vanity_jobs').document(job_id).update({'status': 'RUNNING'})

                        if data.get('notify') and data.get('email'):
                            try:
                                send_start_email(data['email'], job_id, data.get('prefix'), data.get('suffix'), data.get('price_sol'), data.get('is_trial'))
                            except Exception as e:
                                logging.error(f"Failed to send start email: {e}")

                        # Cleanup temp PIN for security BEFORE dispatching
                        db.collection('vanity_jobs').document(job_id).update({'temp_pin': firestore.DELETE_FIELD})

                        # Trigger Cloud Run (AFTER security cleanup)
                        dispatch_cloud_job(
                            job_id,
                            data.get('user_id'),
                            data.get('prefix'),
                            data.get('suffix'),
                            data.get('case_sensitive', True),
                            pin_plain
                        )

                        running_cloud += 1

                except Exception as e:
                    logging.exception(f"❌ Error processing job {doc.id}")
                    try:
                        db.collection('vanity_jobs').document(doc.id).update({'status': 'FAILED', 'error': str(e)})
                    except: pass

        except Exception as e:
            # The database index failure logs here.
            logging.exception(f"Scheduler Error")
        
        # Check queue every 5 seconds
        time.sleep(5)

@app.route('/check-user', methods=['POST'])
@limiter.limit("20 per minute")
def check_user():
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id')

    # Securely verify ID token if provided
    id_token = data.get('id_token')
    if id_token:
        try:
            decoded_token = auth.verify_id_token(id_token)
            verified_email = decoded_token.get('email')
            if verified_email:
                session['email'] = verified_email
                logging.info(f"✅ Verified session for {verified_email}")
        except Exception as e:
            logging.warning(f"❌ Token Verification Failed: {e}")
            # Do NOT set session email if verification fails

    db = None
    try:
        db = firestore.Client(project=PROJECT_ID)
        doc = db.collection('users').document(user_id).get()
        user_data = doc.to_dict() if doc.exists else {}
        return jsonify({
            'has_pin': bool(user_data.get('pin_hash')),
            'trials_used': check_user_trials(user_id),
            'referred_by': user_data.get('referred_by')
        })
    except Exception as e: return jsonify({'error': str(e)}), 500
    finally:
        cleanup_firestore_client(db)

@app.route('/set-pin', methods=['POST'])
@limiter.limit("5 per minute")
def set_pin():
    data = request.get_json(silent=True) or {}
    user_id, pin = data.get('user_id'), data.get('pin')
    db = None
    try:
        hashed = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
        db = firestore.Client(project=PROJECT_ID)
        db.collection('users').document(user_id).set({'user_id': user_id, 'pin_hash': hashed, 'updated_at': firestore.SERVER_TIMESTAMP}, merge=True)
        return jsonify({'message': 'PIN set'})
    except Exception as e: return jsonify({'error': str(e)}), 500
    finally:
        cleanup_firestore_client(db)

# --- REFERRAL ENDPOINTS ---
@app.route('/api/referral/create', methods=['POST'])
@limiter.limit("30 per minute")
def create_referral():
    data = request.get_json(silent=True) or {}
    user_id, pin = data.get('user_id'), data.get('pin')
    if not user_id or not pin: return jsonify({'error': 'Missing ID or PIN'}), 400

    db = None
    try:
        db = firestore.Client(project=PROJECT_ID)

        # Verify PIN
        udoc = db.collection('users').document(user_id).get()
        if not udoc.exists or not udoc.to_dict().get('pin_hash'): return jsonify({'error': 'PIN not set'}), 400
        user_data = udoc.to_dict()
        if not bcrypt.checkpw(pin.encode(), user_data.get('pin_hash').encode()): return jsonify({'error': 'Invalid PIN'}), 401

        # Check if already has code (Idempotency Check)
        ref_doc = db.collection('referrals').document(user_id).get()
        if ref_doc.exists:
            return jsonify(ref_doc.to_dict())

        # Generate Unique Code
        code = generate_referral_code()
        # Ensure uniqueness (simple check, retry loop could be better but collision prob is low for now)
        # In a high volume system, we'd check 'referrals' collection where code == new_code

        new_ref = {
            'user_id': user_id,
            'code': code,
            'balance_sol': 0.0,
            'total_earnings': 0.0,
            'usage_count': 0,
            'created_at': firestore.SERVER_TIMESTAMP
        }
        db.collection('referrals').document(user_id).set(new_ref)
        return jsonify({
            "success": True,
            "code": code,
            "message": "Referral code created successfully"
        })
    except Exception as e:
        logging.exception("Referral Create Error")
        return jsonify({'error': str(e)}), 500
    finally:
        cleanup_firestore_client(db)

@app.route('/api/referral/stats', methods=['GET'])
@limiter.limit("30 per minute")
def referral_stats():
    user_id = request.args.get('user_id')
    if not user_id: return jsonify({'error': 'Missing user_id'}), 400

    db = None
    try:
        db = firestore.Client(project=PROJECT_ID)
        doc = db.collection('referrals').document(user_id).get()
        if not doc.exists: return jsonify({'error': 'No referral account'}), 404
        return jsonify(doc.to_dict())
    except Exception as e: return jsonify({'error': str(e)}), 500
    finally:
        cleanup_firestore_client(db)

@app.route('/api/referral/validate', methods=['POST'])
@limiter.limit("30 per minute")
def validate_referral():
    data = request.get_json(silent=True) or {}
    code = data.get('code', '').strip().upper()
    if not code: return jsonify({'valid': False}), 200

    db = None
    try:
        db = firestore.Client(project=PROJECT_ID)
        # Query for code
        docs = db.collection('referrals').where('code', '==', code).limit(1).stream()
        valid = any(docs)
        return jsonify({'valid': valid, 'discount_percent': 10 if valid else 0})
    except Exception as e: return jsonify({'error': str(e)}), 500
    finally:
        cleanup_firestore_client(db)

@app.route('/api/referral/withdraw', methods=['POST'])
@limiter.limit("30 per minute")
def withdraw_referral():
    data = request.get_json(silent=True) or {}
    user_id, pin = data.get('user_id'), data.get('pin')
    amount = float(data.get('amount', 0))
    target_wallet = data.get('target_wallet')

    if amount < 0.1: return jsonify({'error': 'Min withdrawal 0.1 SOL'}), 400
    if not is_base58(target_wallet): return jsonify({'error': 'Invalid Wallet'}), 400

    db = None
    try:
        db = firestore.Client(project=PROJECT_ID)

        # Verify PIN
        udoc = db.collection('users').document(user_id).get()
        if not udoc.exists or not bcrypt.checkpw(pin.encode(), udoc.to_dict().get('pin_hash').encode()):
            return jsonify({'error': 'Invalid PIN'}), 401

        ref_ref = db.collection('referrals').document(user_id)

        # Transactional update
        @firestore.transactional
        def update_balance(transaction, ref_ref):
            snapshot = transaction.get(ref_ref)
            if not snapshot.exists: raise Exception("No referral account")
            balance = snapshot.get('balance_sol')
            if balance < amount: raise Exception("Insufficient balance")

            transaction.update(ref_ref, {'balance_sol': balance - amount})
            return balance - amount

        transaction = db.transaction()
        new_balance = update_balance(transaction, ref_ref)

        # Create Withdrawal Request
        db.collection('withdrawal_requests').add({
            'user_id': user_id,
            'amount': amount,
            'target_wallet': target_wallet,
            'status': 'PENDING',
            'created_at': firestore.SERVER_TIMESTAMP
        })

        # Notify Admin
        send_email_wrapper(
            ADMIN_EMAILS.copy().pop(), # Just send to one admin
            f"[Ref Liquidation Request] User: {user_id}",
            f"User {user_id} requested {amount} SOL to {target_wallet}. New Balance: {new_balance}"
        )

        return jsonify({'message': 'Withdrawal requested', 'new_balance': new_balance})

    except Exception as e:
        logging.exception("Withdrawal Error")
        return jsonify({'error': str(e)}), 400
    finally:
        cleanup_firestore_client(db)

@app.route('/submit-job', methods=['POST'])
@limiter.limit("60 per minute")
def submit_job():
    data = request.get_json(silent=True) or {}
    user_id = data.get('userId') or data.get('user_id')
    prefix, suffix = data.get('prefix'), data.get('suffix', '')
    pin, tx_sig = data.get('pin'), data.get('transaction_signature')
    email, notify = data.get('email'), data.get('notify', False)
    # use_gpu ignored, 100% GPU now
    referral_code = data.get('referral_code', '').strip().upper()
    
    if not user_id or not pin: return jsonify({'error': 'Missing ID or PIN'}), 400
    if not prefix and not suffix: return jsonify({'error': 'Prefix/Suffix required'}), 400
    if (prefix and not is_base58(prefix)) or (suffix and not is_base58(suffix)): return jsonify({'error': 'Invalid Base58'}), 400

    db = None
    try:
        # 1. Instantiate the client (always fresh)
        db = firestore.Client(project=PROJECT_ID)

        udoc = db.collection('users').document(user_id).get()
        if not udoc.exists or not udoc.to_dict().get('pin_hash'): return jsonify({'error': 'PIN not set'}), 400

        user_data = udoc.to_dict()
        
        # --- ADMIN / GOD MODE DETECTION ---
        is_admin = email in ADMIN_EMAILS
        is_god_mode = user_data.get('god_mode', False) or is_admin

        # --- REFERRAL LOGIC: LIFETIME BINDING ---
        active_referral_code = None
        # 1. Check Lifetime Binding
        if user_data.get('referred_by'):
            active_referral_code = user_data.get('referred_by')
        # 2. If not bound, check provided code
        elif referral_code:
            # Validate code
            ref_docs = list(db.collection('referrals').where('code', '==', referral_code).limit(1).stream())
            if ref_docs:
                referrer_doc = ref_docs[0]
                # Prevent self-referral
                if referrer_doc.id != user_id:
                     # Bind User
                     db.collection('users').document(user_id).update({'referred_by': referral_code})
                     active_referral_code = referral_code

        # --- ACTIVE JOB CHECK ---
        # Admins (God Mode) bypass the active job check to run parallel jobs
        if not is_god_mode:
            active_jobs = db.collection('vanity_jobs') \
                .where('user_id', '==', user_id) \
                .where('status', 'in', ['QUEUED', 'RUNNING']) \
                .limit(1).stream()

            if any(active_jobs):
                return jsonify({'error': 'You have an active job'}), 400

        if not bcrypt.checkpw(pin.encode(), user_data.get('pin_hash').encode()): return jsonify({'error': 'Invalid PIN'}), 401
        
        # --- DETERMINE JOB TYPE ---
        total_len = len(prefix or '') + len(suffix or '')
        
        # Prices Logic
        if total_len <= 4: base = 0.25
        elif total_len == 5: base = 0.50
        elif total_len == 6: base = 1.00
        elif total_len == 7: base = 2.00
        elif total_len == 8: base = 3.00
        else: base = 5.00

        # Note: GPU Surcharge removed as we are 100% GPU now.
        
        # Free logic: <5 chars AND trials remaining OR Admin/God Mode
        if is_admin:
            price = 0.0
        elif total_len <= 4 and check_user_trials(user_id) < 2:
            price = 0.0
        else:
            price = base * 0.5
            # Apply Referral Discount (Extra 10%)
            if active_referral_code:
                price = price * 0.90
        
        # Hard Limit
        if total_len > 8: return jsonify({'error': 'Max 8 chars allowed in Beta'}), 403

        # Payment Check
        if price > 0:
            if not tx_sig: return jsonify({'error': f'Payment of {price} SOL required'}), 402
            if not verify_payment(tx_sig, price): return jsonify({'error': 'Payment verification failed'}), 402

        job_id = str(uuid.uuid4())
        if tx_sig:
             logging.info(f"💰 Job {job_id} Payment: {tx_sig}")
        
        # --- JOB QUEUE LOGIC ---
        # Determine Worker Type & Cloud Status
        worker_type = "cloud-run-gpu-redpanda"

        is_cloud_job = True

        # 2. Save Initial State as 'QUEUED'
        db.collection('vanity_jobs').document(job_id).set({
            'job_id': job_id, 
            'user_id': user_id, 
            'prefix': prefix, 
            'suffix': suffix, 
            'case_sensitive': data.get('case_sensitive', True),
            'status': 'QUEUED', 
            'is_cloud': is_cloud_job,
            'worker_type': worker_type,
            'temp_pin': pin,
            'created_at': firestore.SERVER_TIMESTAMP, 
            'is_trial': (price == 0), 
            'price_sol': price,
            'transaction_signature': tx_sig, 
            'email': email, 
            'notify': notify
        })

        # --- PROCESS COMMISSION (Async/Optimistic) ---
        if active_referral_code and price > 0:
            try:
                # Find referrer (we did this earlier if binding, but safe to query again or cache)
                # If we bound it, we know it exists. If it was already bound, we need to find the doc id.
                # Optimization: Query by code
                ref_docs = list(db.collection('referrals').where('code', '==', active_referral_code).limit(1).stream())
                if ref_docs:
                    referrer_ref = ref_docs[0].reference
                    commission = price * 0.15

                    # Transactional increment
                    @firestore.transactional
                    def add_commission(transaction, ref):
                        snap = transaction.get(ref)
                        if snap.exists:
                            transaction.update(ref, {
                                'balance_sol': snap.get('balance_sol') + commission,
                                'total_earnings': snap.get('total_earnings') + commission,
                                'usage_count': snap.get('usage_count') + 1
                            })

                    transaction = db.transaction()
                    add_commission(transaction, referrer_ref)
                    logging.info(f"💰 Commission {commission} SOL credited to {referrer_ref.id} for job {job_id}")

            except Exception as e:
                logging.error(f"Failed to credit commission: {e}")

        return jsonify({'job_id': job_id, 'message': 'Job Queued successfully'}), 202

    except Exception as e:
        logging.exception("Submit job failed")
        return jsonify({'error': str(e)}), 500

    finally:
        # 3. CRITICAL CLEANUP: Ensure the client connection is closed immediately after use
        cleanup_firestore_client(db)

@app.route('/api/worker/complete', methods=['POST'])
def worker_complete():
    data = request.get_json(silent=True) or {}
    job_id = data.get('job_id')
    public_key = data.get('public_key')
    secret_key = data.get('secret_key')
    error = data.get('error')

    if not job_id: return jsonify({'error': 'Missing job_id'}), 400

    db = None
    try:
        db = firestore.Client(project=PROJECT_ID)
        job_ref = db.collection('vanity_jobs').document(job_id)
        job = job_ref.get()

        if not job.exists:
             return jsonify({'error': 'Job not found'}), 404

        job_data = job.to_dict()
        if job_data.get('status') != 'RUNNING':
             # If it's already completed or failed, we just return success to stop worker retries
             return jsonify({'success': True, 'message': 'Job already processed'})

        if error:
            job_ref.update({'status': 'FAILED', 'error': error})
            logging.error(f"Worker reported error for {job_id}: {error}")
            return jsonify({'success': True})

        if not public_key or not secret_key:
            return jsonify({'error': 'Missing keys'}), 400

        job_ref.update({
            'status': 'COMPLETED',
            'public_key': public_key,
            'secret_key': secret_key,
            'completed_at': firestore.SERVER_TIMESTAMP
        })
        logging.info(f"Job {job_id} completed successfully via Worker Callback.")

        if job_data.get('notify') and job_data.get('email'):
            send_completion_email(job_data['email'], public_key)

        return jsonify({'success': True})

    except Exception as e:
        logging.exception("Worker Complete Error")
        return jsonify({'error': str(e)}), 500
    finally:
        cleanup_firestore_client(db)


@app.route('/reveal-key', methods=['POST'])
@limiter.limit("5 per minute")
def reveal_key():
    data = request.get_json(silent=True) or {}
    job_id, pin = data.get('job_id'), data.get('pin')
    db = None
    try:
        db = firestore.Client(project=PROJECT_ID)
        job = db.collection('vanity_jobs').document(job_id).get()
        if not job.exists: return jsonify({'error': 'Job not found'}), 404
        jdata = job.to_dict()
        
        # --- CRITICAL SECURITY FIX ---
        user_doc = db.collection('users').document(jdata['user_id']).get() 
        pin_hash = user_doc.to_dict().get('pin_hash', '') if user_doc.exists else ''

        # If the hash exists and the provided PIN does NOT match the hash, reject.
        if pin_hash and not bcrypt.checkpw(pin.encode(), pin_hash.encode()):
            return jsonify({'error': 'Invalid PIN'}), 401
        
        # PIN is correct for the job owner, proceed to decryption.
        key = generate_key_from_pin(pin, jdata['user_id'])
        return jsonify({'secret_key': Fernet(key).decrypt(jdata['secret_key'].encode()).decode()})
    except Exception as e: 
        logging.exception("Decryption Error")
        return jsonify({'error': 'Decryption failed'}), 400
    finally:
        cleanup_firestore_client(db)

@app.route('/api/check-sns', methods=['POST'])
def check_sns():
    """Proxy route to check SNS availability via Mainnet RPC"""
    data = request.get_json(silent=True) or {}
    public_key = data.get('publicKey')

    if not public_key:
        return jsonify({'error': 'Missing publicKey'}), 400

    # Payload for getAccountInfo
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [
            public_key,
            {"encoding": "base58"}
        ]
    }

    try:
        # Use the standard mainnet-beta endpoint as requested
        # Server-side request avoids browser CORS issues
        rpc_url = "https://api.mainnet-beta.solana.com"
        resp = requests.post(rpc_url, json=payload, timeout=5)

        # If the RPC returns an error (even 200 OK with error body), forward it or handle it
        if resp.status_code != 200:
            logging.error(f"SNS Proxy RPC Error: {resp.status_code} {resp.text}")
            return jsonify({'error': 'RPC Error'}), resp.status_code

        rpc_data = resp.json()
        return jsonify(rpc_data)

    except Exception as e:
        logging.exception("SNS Proxy Failed")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    # START THE TRAFFIC CONTROLLER THREAD
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()
    app.run(host='127.0.0.1', port=8080)
