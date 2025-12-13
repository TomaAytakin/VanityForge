// utils/sns.js

(function(window) {
    // Constants for Solana Name Service
    const SNS_PROGRAM_ID = new solanaWeb3.PublicKey('namesLPneVptA9Z5rqUDD9tMTWEJwofgaYwp8cawRkX');
    const ROOT_DOMAIN_ACCOUNT = new solanaWeb3.PublicKey('58PwtjSDuFHuUkYjH9BYnnQKHfwo9reZhC2zMJv9JPkx');
    const HASH_PREFIX = 'SPL Name Service';

    // Helper to hash the name with the prefix
    async function getHashedName(name) {
        const input = HASH_PREFIX + name;
        const encoder = new TextEncoder();
        const data = encoder.encode(input);
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        return new Uint8Array(hashBuffer);
    }

    // Helper to derive the domain key
    // Logic from @bonfida/spl-name-service:
    // seeds = [hashed_name, class (or 32 zeros), parent (or 32 zeros)]
    async function getDomainKey(domainName) {
        // Strip .sol if present
        let name = domainName;
        if (name.endsWith('.sol')) {
            name = name.slice(0, -4);
        }

        const hashedName = await getHashedName(name);

        // nameClass is usually empty (zeros) for standard .sol domains
        const nameClass = new Uint8Array(32);

        // parent is ROOT_DOMAIN_ACCOUNT for .sol domains
        const parent = ROOT_DOMAIN_ACCOUNT.toBuffer();

        const seeds = [
            hashedName,
            nameClass,
            parent
        ];

        const [key] = await solanaWeb3.PublicKey.findProgramAddress(
            seeds,
            SNS_PROGRAM_ID
        );
        return key;
    }

    // Main function to check availability
    async function checkDomainAvailability(domainName) {
        try {
            const domainKey = await getDomainKey(domainName);

            // Use local proxy to avoid CORS issues with api.mainnet-beta.solana.com
            const response = await fetch('/api/check-sns', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ publicKey: domainKey.toBase58() })
            });

            if (!response.ok) {
                throw new Error(`Proxy error: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error.message || "RPC Error");
            }

            // data.result.value is null if account doesn't exist (Available)
            // It is an object if it exists (Taken)
            const accountInfo = data.result ? data.result.value : null;

            // If accountInfo is null, the domain is available
            return accountInfo === null;

        } catch (e) {
            console.error("Error checking domain availability:", e);
            throw e;
        }
    }

    // Expose via global object
    window.SNSUtils = {
        checkDomainAvailability,
        getDomainKey
    };

})(window);
