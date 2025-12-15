# VanityForge: The Apex of Crypto Identity

![VanityForge Main Banner](https://github.com/TomaAytakin/VanityForge/blob/main/assets/readmeherobanner.png)

Welcome to **VanityForge**. You have arrived at the *absolute pinnacle* of Solana address generation. We provide what others cannot: **Unmatched Speed**, <u>Military-Grade Security</u>, and **24/7 Reliability**.

Stop settling for random addresses. Demand a customized identity that commands *respect* on the blockchain.

---

![Solana Mainnet](https://img.shields.io/badge/Network-Solana_Mainnet-green?style=flat-square&logo=solana)
![Bonfida SNS](https://img.shields.io/badge/Integration-Bonfida_SNS-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)
![Security](https://img.shields.io/badge/Security-Audit_Ready-shield?style=flat-square)

## 🛡️ Architecture & Security

VanityForge prioritizes user safety through a "Glass Box" engineering philosophy. Unlike standard dApps that rely on opaque SDK methods, we employ a custom **Transaction Safety Engine**.

### 🔒 Manual Transaction Composition
We do not use "black box" signing methods. Every transaction is manually composed:
1.  **Instruction Sanitization:** All SNS SDK instructions are intercepted, flattened, and inspected for correct program IDs before being added to the transaction.
2.  **Explicit Fee Payer:** The user's wallet is explicitly set as the fee payer and authority, preventing unauthorized delegation.
3.  **Manual Signing Flow:** We use `signTransaction` (Manual) instead of `signAndSend` to ensure the user has full visibility into the transaction payload before broadcasting.

### ⚡ Direct-to-RPC Injection (Helius)
To prevent rate-limiting and 403 errors common with public RPCs:
* We utilize **Helius Premium RPCs**.
* The endpoint is securely injected from our Python backend directly into the frontend session.
* This ensures high-speed transaction delivery (`confirmed` commitment) without exposing API keys in the client source code unnecessarily.

### 🔍 Radical Transparency
* **Open Source:** All transaction logic is located in `utils/sns.js` and is un-obfuscated.
* **Standard Protocols:** We strictly integrate with the official `@bonfida/spl-name-service` and `@solana/web3.js`.

## 🛠️ Tech Stack

*   **Frontend:** HTML5, TailwindCSS, Vanilla JS (No frameworks for maximum auditability).
*   **Backend:** Python 3 (Flask), Google Cloud Run.
*   **Blockchain:** Solana Web3.js v1.x, Bonfida SNS SDK.

## 🗺️ Project Roadmap

### ✅ Completed
*   **Core Vanity Engine (GPU/CPU)**
*   **Solana Name Service (SNS) Integration**
*   **High-Performance RPC Integration (Helius)**
*   **Transaction Safety Engine (Manual Signing Flow)**

### 🚧 In Progress
*   **Security Audits & Whitelisting (Blowfish/Phantom)**
*   **Mobile UI Optimization**

### 🚀 Upcoming
*   **Leaderboard & Analytics**
*   **Official V1 Public Launch**

_____________________________________________________________________________________
*Secure your legacy on Solana today with VanityForge.*
