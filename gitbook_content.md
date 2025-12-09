# User Manual: Getting Started

## Connecting Your Wallet & Authentication
VanityForge uses a hybrid authentication model combining Web3 anonymity with Web2 convenience.

1.  **Connect Phantom Wallet:**
    *   Navigate to the homepage and locate the **"Connect Phantom"** button (Button ID: `#connect-phantom-btn`) in the center of the screen.
    *   Clicking this triggers a connection request to your browser's Phantom extension.
    *   Approve the request to allow VanityForge to read your public key. This is required for identifying your account and processing Solana payments.
    *   *Note:* If you do not have Phantom installed, the button will open `https://phantom.app/` in a new tab.

2.  **Login with Google (Optional but Recommended):**
    *   To enable email notifications and account recovery features, click **"Login with Google"**.
    *   This opens a secure Firebase pop-up window (`firebase.auth().signInWithPopup`).
    *   Once authenticated, your Google User ID is linked to your session.

## Creating Your Non-Recoverable Encryption PIN
Upon your first successful login, the system checks the `users` database. If no PIN hash is found, you are immediately presented with the **Vault PIN Creation Modal**.

*   **The Warning:** You will see a strict warning box:
    > "You are about to create a permanent encryption PIN. VanityForge does not store this PIN. If you lose it, you will permanently lose the capability to reveal your private keys or forge new wallets within your VanityForge account. We cannot recover it for you."
*   **The Acknowledgment:** You **must** manually check the confirmation box:
    > "I acknowledge that VanityForge cannot recover my PIN if lost, and that losing my PIN means I will be unable to reveal my private keys or access my VanityForge account functions."
*   **Setting the PIN:** Enter a 4-6 digit numeric PIN using the on-screen secure numpad. This PIN is hashed via `bcrypt` before reaching our database; the raw PIN is never stored.

---

# User Manual: Submitting a Forge Job

## Understanding Difficulty & Time
The time required to find a vanity address increases exponentially with every character you add. We use the formula $58^L$ (where $L$ is length) to estimate difficulty based on the Base58 character set.

*   **1-4 Characters:** Usually "Instant" (< 1 second). Free for trial users.
*   **5 Characters:** ~1-2 minutes.
*   **6 Characters:** ~1-2 hours.
*   **7 Characters:** ~2-3 days (Requires Cloud Grinding).
*   **8 Characters:** Weeks/Months (Extreme difficulty).

*Note:* If your estimated time exceeds 1 hour, the UI will display a browser confirmation dialog asking if you accept the wait time before proceeding.

## Job Status Lifecycle
Once submitted, your job moves through the following statuses in the `vanity_jobs` collection:

1.  **QUEUED:** The job has been accepted by the API and is waiting in the `vanity_jobs` database. The `scheduler_loop` checks this queue every 5 seconds.
2.  **RUNNING:**
    *   **Local Jobs (<5 chars):** Running on the main server's CPU using `multiprocessing`.
    *   **Cloud Jobs (5+ chars):** Dispatched to **Google Cloud Run** (`vanity-gpu-worker`). The system updates the status to RUNNING only after successfully dispatching the container.
3.  **COMPLETED:** The vanity address was found! The private key has been encrypted with your PIN-derived key and saved. An email notification (if enabled) is sent.
4.  **FAILED:** The job encountered an error (e.g., Cloud Dispatch failure, process crash). The specific error message is saved in the `error` field.

---

# Glossary & FAQ

## Terminology
*   **Solana Base58:** The specific character set used for Solana addresses. It includes alphanumeric characters but **excludes** `0` (Zero), `O` (Capital o), `I` (Capital i), and `l` (Lower L) to avoid visual confusion.
*   **Vanity Address:** A cryptocurrency wallet address that starts (Prefix) or ends (Suffix) with a specific sequence of characters chosen by the user (e.g., `VaNiTy...`).
*   **Encryption PIN:** A user-defined numeric code used to generate the AES-128 (Fernet) key that encrypts your private wallet keys.
*   **Service Account:** A special Google Cloud account used by the backend to authenticate against Firestore and Cloud Run without human intervention.
*   **Gas/SOL Fees:** The transaction cost required by the Solana network to process payments to the VanityForge Treasury.

## Frequently Asked Questions

**Q: If I lose my PIN, is my wallet lost?**
**A:** **Yes and No.**
*   **The Keys:** You will **permanently lose access** to reveal the private key stored on VanityForge. Since we do not store your raw PIN, we cannot decrypt the key for you.
*   **The Wallet:** If you have already revealed the key and saved it elsewhere (e.g., in Phantom), your wallet is safe. The PIN only protects the *backup* stored on our server.

**Q: Why do I receive emails from "Support" when the admin logs in?**
**A:** This is a security feature. The system uses a centralized SMTP configuration. Even though the backend authenticates with the `SMTP_EMAIL` (e.g., `admin@`), the `send_email_wrapper` function explicitly sets the email header to:
`From: VanityForge Support <support@vanityforge.org>`
This ensures professional branding and prevents personal admin email addresses from being exposed to users.

---

# System Architecture Deep Dive (Extravagant Details)

## Server Environment & Stability
The core orchestrator (`vm_server.py`) runs on a high-performance Google Compute Engine VM. Stability is enforced via `systemd` and `Gunicorn`.

*   **Process Management:** The application runs as a `systemd` service, ensuring it automatically restarts on boot or failure.
*   **Resource Limits (`LimitNOFILE`):**
    To handle high-concurrency Firestore connections and subprocess management, the systemd unit file is configured with `LimitNOFILE=65535`. This prevents "Too many open files" errors which can occur when the `scheduler_loop` spawns multiple `multiprocessing.Process` workers or maintains open socket connections to Google Cloud APIs.
*   **Zombie Reaper:** The application explicitly handles zombie processes using `signal.signal(signal.SIGCHLD, signal.SIG_IGN)` in the main entry point, preventing defunct processes from consuming the process table.

## Database Structure (Firestore)
The database is strictly partitioned into two primary collections to separate user credentials from job data.

### 1. `users` Collection
Stores authentication and security state.
*   `user_id` (String): The unique Google UID or Phantom Public Key.
*   `pin_hash` (String): The `bcrypt` hash of the user's PIN.
*   `updated_at` (Timestamp): Last modification time.
*   *Note regarding God Mode:* Unlike standard users, "God Mode" status is **not** stored as a boolean in the database. Instead, it is dynamically determined at runtime by checking if the user's email exists in the hardcoded `ADMIN_EMAILS` set in `vm_server.py`.

### 2. `vanity_jobs` Collection
Stores the state of every forging attempt.
*   `job_id` (UUID String): Unique identifier for the job.
*   `user_id` (String): Link to the owner.
*   `status` (String): `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`.
*   `prefix` / `suffix` (String): The target patterns.
*   `temp_pin` (String): **Transient Field.** Briefly stores the raw PIN during the `QUEUED` phase to allow the worker to initialize encryption. This field is explicitly deleted (`firestore.DELETE_FIELD`) immediately upon job dispatch.
*   `secret_key` (String): The encrypted private key (only present after `COMPLETED`).
*   `public_key` (String): The resulting wallet address.
*   `active_job_id`: **Virtual Association.** The system does not store an `active_job_id` field in the user document. Instead, active status is determined by querying this collection for any documents where `user_id == current_user` AND `status` is `QUEUED` or `RUNNING`.
