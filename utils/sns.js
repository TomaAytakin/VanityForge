// utils/sns.js

(function(window) {
    // Constants for Solana Name Service
    const SNS_PROGRAM_ID = new solanaWeb3.PublicKey('namesLPneVptA9Z5rqUDD9tMTWEJwofgaYwp8cawRkX');
    const ROOT_DOMAIN_ACCOUNT = new solanaWeb3.PublicKey('58PwtjSDuFHuUkYjH9BYnnQKHfwo9reZhC2zMJv9JPkx');
    const WRAPPED_SOL_MINT = new solanaWeb3.PublicKey('So11111111111111111111111111111111111111112');
    const TOKEN_PROGRAM_ID = new solanaWeb3.PublicKey('TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA');
    const HASH_PREFIX = 'SPL Name Service';
    const REFERRER_KEY = new solanaWeb3.PublicKey("CUfjsGUee8u83dfFxHt1jXJUCRLiF1KoWVYcKyVforGe");

    // Helper to hash the name with the prefix
    async function getHashedName(name) {
        const input = HASH_PREFIX + name;
        const encoder = new TextEncoder();
        const data = encoder.encode(input);
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        return new Uint8Array(hashBuffer);
    }

    async function getDomainKey(domainName) {
        let name = domainName;
        if (name.endsWith('.sol')) {
            name = name.slice(0, -4);
        }
        const hashedName = await getHashedName(name);
        const nameClass = new Uint8Array(32);
        const parent = ROOT_DOMAIN_ACCOUNT.toBuffer();
        const seeds = [hashedName, nameClass, parent];
        const [key] = await solanaWeb3.PublicKey.findProgramAddress(seeds, SNS_PROGRAM_ID);
        return key;
    }

    async function checkDomainAvailability(domainName) {
        try {
            const domainKey = await getDomainKey(domainName);
            const response = await fetch('/api/check-sns', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ publicKey: domainKey.toBase58() })
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
     * Wraps SOL to wSOL automatically.
     * Includes Referrer Key for commission.
     */
    async function createRegisterTransaction(connection, wallet, domainName, space = 1000) {
        if (!window.registerDomainNameV2) {
            throw new Error("Bonfida SDK not loaded.");
        }

        // Clean domain name
        let name = domainName.toLowerCase();
        if (name.endsWith('.sol')) name = name.slice(0, -4);

        const buyer = new solanaWeb3.PublicKey(wallet.publicKey);

        // 1. Calculate Price
        // Pricing logic: 1 char = $750, 2 = $750, 3 = $150, 4 = $150, 5+ = $20
        // We need the LAMPORTS amount.
        // Usually we should fetch price via pyth or oracle, but Bonfida instruction does that on-chain.
        // We just need to fund the wSOL account with enough SOL.
        // For safety, we can fund with a safe buffer or fetch estimation?
        // Let's use getTvAmount from SDK if available, or manual logic.

        let priceUsd = 20;
        const len = name.length;
        if (len === 1) priceUsd = 750;
        else if (len === 2) priceUsd = 750;
        else if (len === 3) priceUsd = 150;
        else if (len === 4) priceUsd = 150;

        // Convert USD to SOL. We need a price feed.
        // Since we can't easily get price feed here without an API, we might rely on the user having enough.
        // BUT wait, we need to TRANSFER specific amount to wSOL account.
        // If we transfer too little, tx fails.
        // If we transfer too much, we get it back when we close account?
        // Yes, closing the wSOL account returns balance to buyer.
        // So we can overestimate.
        // Assume SOL = $200 (conservative low for calculating SOL needed? No, low SOL price means MORE SOL needed).
        // Assume SOL = $10. Then $750 = 75 SOL.
        // Assume SOL = $1000. Then $750 = 0.75 SOL.
        // Current SOL price is ~$150.
        // Let's add a robust buffer.
        // 1 char: Need ~$750. Say 10 SOL.
        // 5+ chars: Need ~$20. Say 0.5 SOL.

        let solToWrap = 0.2; // Default for 5+ chars (approx $30 at $150/SOL)
        if (len <= 2) solToWrap = 10;
        else if (len <= 4) solToWrap = 2;

        // 2. Create Ephemeral wSOL Account
        const wsolKeypair = solanaWeb3.Keypair.generate();
        const wsolAccount = wsolKeypair.publicKey;

        const rentExempt = await connection.getMinimumBalanceForRentExemption(165); // Token Account size
        const lamportsToTransfer = Math.ceil(solToWrap * solanaWeb3.LAMPORTS_PER_SOL);

        const transaction = new solanaWeb3.Transaction();

        // 3. Create & Fund wSOL Account
        transaction.add(
            solanaWeb3.SystemProgram.createAccount({
                fromPubkey: buyer,
                newAccountPubkey: wsolAccount,
                space: 165,
                lamports: rentExempt + lamportsToTransfer,
                programId: TOKEN_PROGRAM_ID,
            })
        );

        // 4. Initialize wSOL Account
        transaction.add(
            solanaWeb3.TransactionInstruction.from({
                keys: [
                    { pubkey: wsolAccount, isSigner: false, isWritable: true },
                    { pubkey: WRAPPED_SOL_MINT, isSigner: false, isWritable: false },
                    { pubkey: buyer, isSigner: false, isWritable: false }, // Owner
                    { pubkey: solanaWeb3.SYSVAR_RENT_PUBKEY, isSigner: false, isWritable: false },
                ],
                programId: TOKEN_PROGRAM_ID,
                data: Buffer.from([1]), // InitializeAccount instruction (1)
            })
        );

        // 5. Register Domain
        // Signature: registerDomainNameV2(name, space, buyer, buyerTokenAccount, mint?, referrerKey?)
        // Note: SDK might return [ix] or just ix.
        const registerIx = await window.registerDomainNameV2(
            name,
            space,
            buyer,
            wsolAccount,
            WRAPPED_SOL_MINT,
            REFERRER_KEY
        );

        // The SDK might return an array or single instruction.
        if (Array.isArray(registerIx)) {
            registerIx.forEach(ix => transaction.add(ix));
        } else {
            transaction.add(registerIx);
        }

        // 6. Close wSOL Account (Refund remaining SOL)
        transaction.add(
            solanaWeb3.TransactionInstruction.from({
                keys: [
                    { pubkey: wsolAccount, isSigner: false, isWritable: true },
                    { pubkey: buyer, isSigner: false, isWritable: true }, // Destination
                    { pubkey: buyer, isSigner: true, isWritable: false }, // Owner (Signer)
                ],
                programId: TOKEN_PROGRAM_ID,
                data: Buffer.from([9]), // CloseAccount instruction (9)
            })
        );

        // Add signer (wSOL keypair)
        transaction.partialSign(wsolKeypair);

        return transaction;
    }

    // Expose via global object
    window.SNSUtils = {
        checkDomainAvailability,
        getDomainKey,
        createRegisterTransaction
    };

})(window);
