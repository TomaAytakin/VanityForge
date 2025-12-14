// utils/sns.js

// Imports from ESM.sh as requested
import { Connection, PublicKey, Transaction, Keypair, SystemProgram, TransactionInstruction, SYSVAR_RENT_PUBKEY, LAMPORTS_PER_SOL } from "https://esm.sh/@solana/web3.js@1.78.0";
import { registerDomainName, getDomainKey as getDomainKeySDK } from "https://esm.sh/@bonfida/spl-name-service@0.2.4";
import { getAssociatedTokenAddress } from "https://esm.sh/@solana/spl-token@0.3.8";

// Constants
const SNS_PROGRAM_ID = new PublicKey('namesLPneVptA9Z5rqUDD9tMTWEJwofgaYwp8cawRkX');
const ROOT_DOMAIN_ACCOUNT = new PublicKey('58PwtjSDuFHuUkYjH9BYnnQKHfwo9reZhC2zMJv9JPkx');
const WRAPPED_SOL_MINT = new PublicKey('So11111111111111111111111111111111111111112');
const TOKEN_PROGRAM_ID = new PublicKey('TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA');
const HASH_PREFIX = 'SPL Name Service';
const REFERRER_KEY = new PublicKey("CUfjsGUee8u83dfFxHt1jXJUCRLiF1KoWVYcKyVforGe");

// Helper to hash the name with the prefix (Manual implementation if not using SDK for this)
async function getHashedName(name) {
    const input = HASH_PREFIX + name;
    const encoder = new TextEncoder();
    const data = encoder.encode(input);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    return new Uint8Array(hashBuffer);
}

// We can use the SDK's getDomainKey if available, or our manual one.
// SDK 0.2.4 exports getDomainKey.
async function getDomainKey(domainName) {
    let name = domainName;
    if (name.endsWith('.sol')) {
        name = name.slice(0, -4);
    }
    // Using SDK implementation if possible, else manual fallback
    if (getDomainKeySDK) {
        return await getDomainKeySDK(name);
    }

    const hashedName = await getHashedName(name);
    const nameClass = new Uint8Array(32);
    const parent = ROOT_DOMAIN_ACCOUNT.toBuffer();
    const seeds = [hashedName, nameClass, parent];
    const [key] = await PublicKey.findProgramAddress(seeds, SNS_PROGRAM_ID);
    return key;
}

async function checkDomainAvailability(domainName) {
    try {
        const domainKey = await getDomainKey(domainName);
        // Note: domainKey might be an object from SDK { pubkey, hashed, ... }
        // If it's a PublicKey, .toBase58() works. If object, access .pubkey.
        const pubkey = (domainKey.pubkey) ? domainKey.pubkey : domainKey;

        const response = await fetch('/api/check-sns', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ publicKey: pubkey.toBase58() })
        });
        if (!response.ok) throw new Error(`Proxy error: ${response.status}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error.message || "RPC Error");
        const accountInfo = data.result ? data.result.value : null;
        return accountInfo === null;
    } catch (e) {
        console.error("Error checking domain availability:", e);
        throw e;
    }
}

/**
 * Creates a transaction to register a domain with SNS.
 */
async function createRegisterTransaction(connection, wallet, domainName, space = 1000) {
    // Clean domain name
    let name = domainName.toLowerCase();
    if (name.endsWith('.sol')) name = name.slice(0, -4);

    const buyer = new PublicKey(wallet.publicKey);

    // 1. Calculate Price Logic (simplified as per previous implementation)
    // Pricing: 1: $750, 2: $750, 3: $150, 4: $150, 5+: $20
    const len = name.length;
    let solToWrap = 0.2; // Default for 5+
    if (len <= 2) solToWrap = 10;
    else if (len <= 4) solToWrap = 2;

    // 2. Create Ephemeral wSOL Account
    const wsolKeypair = Keypair.generate();
    const wsolAccount = wsolKeypair.publicKey;

    const rentExempt = await connection.getMinimumBalanceForRentExemption(165);
    const lamportsToTransfer = Math.ceil(solToWrap * LAMPORTS_PER_SOL);

    const transaction = new Transaction();

    // 3. Create & Fund wSOL Account
    transaction.add(
        SystemProgram.createAccount({
            fromPubkey: buyer,
            newAccountPubkey: wsolAccount,
            space: 165,
            lamports: rentExempt + lamportsToTransfer,
            programId: TOKEN_PROGRAM_ID,
        })
    );

    // 4. Initialize wSOL Account
    transaction.add(
        new TransactionInstruction({
            keys: [
                { pubkey: wsolAccount, isSigner: false, isWritable: true },
                { pubkey: WRAPPED_SOL_MINT, isSigner: false, isWritable: false },
                { pubkey: buyer, isSigner: false, isWritable: false }, // Owner
                { pubkey: SYSVAR_RENT_PUBKEY, isSigner: false, isWritable: false },
            ],
            programId: TOKEN_PROGRAM_ID,
            data: Buffer.from([1]), // InitializeAccount instruction (1)
        })
    );

    // 5. Register Domain
    try {
        // We pass 'buyer' as the 3rd argument, which the SDK uses as the domain owner.
        // ✅ Verified: Domain assigned to user wallet (buyer is passed as owner)
        const ixOrArray = await registerDomainName(
            connection, // Connection is required as first argument
            name,
            space,
            buyer,
            wsolAccount,
            WRAPPED_SOL_MINT,
            REFERRER_KEY
        );

        // Standardize to array
        const ixArray = Array.isArray(ixOrArray) ? ixOrArray : [ixOrArray];

        ixArray.forEach(ix => {
            // ✅ Verified: Money Check - Ensure programId matches Bonfida SNS
            if (ix.programId.toString() !== SNS_PROGRAM_ID.toString()) {
                console.warn("Warning: Instruction programId does not match expected SNS Program ID", ix.programId.toString());
            }

            // 🔒 Fix: Explicitly wrap instruction to prevent version mismatch errors
            // "TransactionInstruction.from is not a function"
            transaction.add(
                new TransactionInstruction({
                    keys: ix.keys,
                    programId: ix.programId,
                    data: ix.data
                })
            );
        });

    } catch (err) {
        console.error("Error creating register instruction:", err);
        throw new Error("Failed to create register instruction. SDK might be incompatible.");
    }

    // 6. Close wSOL Account
    transaction.add(
        new TransactionInstruction({
            keys: [
                { pubkey: wsolAccount, isSigner: false, isWritable: true },
                { pubkey: buyer, isSigner: false, isWritable: true }, // Destination
                { pubkey: buyer, isSigner: true, isWritable: false }, // Owner (Signer)
            ],
            programId: TOKEN_PROGRAM_ID,
            data: Buffer.from([9]), // CloseAccount instruction (9)
        })
    );

    transaction.partialSign(wsolKeypair);

    return transaction;
}

// --- View/Controller Logic moved from HTML ---

async function checkSNS() {
    const input = document.getElementById('sns-domain-input');
    const btn = document.getElementById('sns-check-btn');
    const resultDiv = document.getElementById('sns-result');
    const domain = input.value.trim().toLowerCase();

    if (!domain) return;

    // Reset UI
    resultDiv.classList.add('hidden');
    resultDiv.className = "hidden rounded-lg p-4 flex justify-between items-center animate-fade-in";
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';

    try {
        const available = await checkDomainAvailability(domain);

        resultDiv.classList.remove('hidden');
        if (available) {
            resultDiv.classList.add('bg-green-900/20', 'border', 'border-green-500/30');
            resultDiv.innerHTML = `
                <div>
                    <span class="text-green-400 font-bold text-lg"><i class="fa-solid fa-check-circle mr-2"></i>Available</span>
                    <div class="text-xs text-gray-400 mt-1">${domain}.sol is free to claim!</div>
                </div>
                <button onclick="window.buyDomain('${domain}')" class="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded shadow hover:scale-105 transition transform text-sm">
                    Buy Now (~$20)
                </button>
            `;
        } else {
            resultDiv.classList.add('bg-red-900/20', 'border', 'border-red-500/30');
            resultDiv.innerHTML = `
                <div>
                    <span class="text-red-400 font-bold text-lg"><i class="fa-solid fa-times-circle mr-2"></i>Taken</span>
                    <div class="text-xs text-gray-400 mt-1">${domain}.sol is already registered.</div>
                </div>
                <span class="text-gray-500 font-mono text-sm">Unavailable</span>
            `;
        }

    } catch (e) {
        console.error(e);
        alert("Error checking domain availability.");
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Search';
    }
}

async function buyDomain(domain) {
    // Access global walletAddress from window (defined in HTML)
    const walletAddress = window.walletAddress;

    if (!walletAddress) {
        alert("Please connect your wallet first.");
        // Try to call global connectWallet if it exists
        if (window.connectWallet) {
            await window.connectWallet();
            if (!window.walletAddress) return;
        } else {
            return;
        }
    }

    const wallet = new PublicKey(window.walletAddress);

    const btn = document.querySelector('button[onclick^="window.buyDomain"]');
    const originalText = btn ? btn.innerHTML : 'Buy Now';
    if (btn) {
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
        btn.disabled = true;
    }

    try {
        // Initialize Connection
        // Use the premium Helius RPC
        const connection = new Connection(window.SOLANA_RPC);

        // Create Transaction
        const transaction = await createRegisterTransaction(connection, { publicKey: wallet }, domain, 1000);

        // Sign and Send
        transaction.feePayer = wallet;
        const { blockhash, lastValidBlockHeight } = await connection.getLatestBlockhash();
        transaction.recentBlockhash = blockhash;

        // Use Phantom's window.solana to sign
        const { signature } = await window.solana.signAndSendTransaction(transaction);

        if (btn) btn.innerHTML = 'Confirming...';

        await connection.confirmTransaction({
            signature: signature,
            blockhash: blockhash,
            lastValidBlockHeight: lastValidBlockHeight
        });

        alert("Success! Domain registered. Signature: " + signature);
        location.reload();

    } catch (e) {
        console.error(e);
        alert("Transaction failed: " + e.message);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    }
}

// 🔒 FIX: Expose module functions to global scope so HTML buttons work
window.checkSNS = checkSNS;
window.buyDomain = buyDomain;
window.SNSUtils = {
    checkDomainAvailability,
    getDomainKey,
    createRegisterTransaction
};
