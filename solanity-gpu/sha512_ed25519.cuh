/* sha512_ed25519.cuh - Specialized for 32-byte inputs (Ed25519 seeds) */
#ifndef SHA512_ED25519_CUH
#define SHA512_ED25519_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// Standard SHA-512 Constants (K)
__constant__ uint64_t K_SHA512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

#define GPU_ROR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
#define GPU_Ch(x, y, z) ((x & y) ^ (~x & z))
#define GPU_Maj(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define GPU_Sigma0(x) (GPU_ROR64(x, 28) ^ GPU_ROR64(x, 34) ^ GPU_ROR64(x, 39))
#define GPU_Sigma1(x) (GPU_ROR64(x, 14) ^ GPU_ROR64(x, 18) ^ GPU_ROR64(x, 41))
#define GPU_Gamma0(x) (GPU_ROR64(x, 1) ^ GPU_ROR64(x, 8) ^ (x >> 7))
#define GPU_Gamma1(x) (GPU_ROR64(x, 19) ^ GPU_ROR64(x, 61) ^ (x >> 6))

// Simplified Load Big-Endian
__device__ __forceinline__ uint64_t load64be(const unsigned char* p) {
    return ((uint64_t)p[0] << 56) | ((uint64_t)p[1] << 48) |
           ((uint64_t)p[2] << 40) | ((uint64_t)p[3] << 32) |
           ((uint64_t)p[4] << 24) | ((uint64_t)p[5] << 16) |
           ((uint64_t)p[6] << 8) | (uint64_t)p[7];
}

// Store Big-Endian
__device__ __forceinline__ void store64be(uint64_t v, unsigned char* p) {
    p[0] = (v >> 56) & 0xFF; p[1] = (v >> 48) & 0xFF;
    p[2] = (v >> 40) & 0xFF; p[3] = (v >> 32) & 0xFF;
    p[4] = (v >> 24) & 0xFF; p[5] = (v >> 16) & 0xFF;
    p[6] = (v >> 8) & 0xFF;  p[7] = v & 0xFF;
}

// Specialized SHA512 for exactly 32 bytes of input
// No loops, no context structs, minimal register pressure
__device__ void sha512_32byte_seed(const unsigned char* seed, unsigned char* out_hash) {
    // 1. Initial State
    uint64_t a = 0x6a09e667f3bcc908ULL;
    uint64_t b = 0xbb67ae8584caa73bULL;
    uint64_t c = 0x3c6ef372fe94f82bULL;
    uint64_t d = 0xa54ff53a5f1d36f1ULL;
    uint64_t e = 0x510e527fade682d1ULL;
    uint64_t f = 0x9b05688c2b3e6c1fULL;
    uint64_t g = 0x1f83d9abfb41bd6bULL;
    uint64_t h = 0x5be0cd19137e2179ULL;

    uint64_t W[16];

    // 2. Prepare Message Schedule (W)
    // Block 0: 32 bytes seed + 0x80 padding + zeros + length
    // Words 0-3: The Seed (32 bytes)
    W[0] = load64be(seed + 0);
    W[1] = load64be(seed + 8);
    W[2] = load64be(seed + 16);
    W[3] = load64be(seed + 24);

    // Words 4-15: Padding
    // Byte 32 is 0x80. Next 7 bytes are 0x00.
    W[4] = 0x8000000000000000ULL;
    W[5] = 0; W[6] = 0; W[7] = 0; W[8] = 0;
    W[9] = 0; W[10] = 0; W[11] = 0; W[12] = 0;
    W[13] = 0; W[14] = 0;

    // Word 15: Length in bits (256 bits = 32 bytes * 8)
    W[15] = 256;

    // 3. The Compression Loop (Unrolled into 80 rounds)
    // We use a rolling window to avoid storing W[16..79]
    uint64_t T1, T2;

    #pragma unroll
    for (int t = 0; t < 80; t++) {
        // Calculate W for t >= 16 on the fly or reuse buffer
        // Since we are register constrained, we recycle W array using mask
        if (t >= 16) {
            W[t & 15] = GPU_Gamma1(W[(t - 2) & 15]) + W[(t - 7) & 15] +
                        GPU_Gamma0(W[(t - 15) & 15]) + W[(t - 16) & 15];
        }

        T1 = h + GPU_Sigma1(e) + GPU_Ch(e, f, g) + K_SHA512[t] + W[t & 15];
        T2 = GPU_Sigma0(a) + GPU_Maj(a, b, c);

        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    // 4. Final Add (No need to update state, just add to initial constants)
    store64be(0x6a09e667f3bcc908ULL + a, out_hash + 0);
    store64be(0xbb67ae8584caa73bULL + b, out_hash + 8);
    store64be(0x3c6ef372fe94f82bULL + c, out_hash + 16);
    store64be(0x510e527fade682d1ULL + d, out_hash + 24);
    store64be(0x510e527fade682d1ULL + e, out_hash + 32);
    store64be(0x9b05688c2b3e6c1fULL + f, out_hash + 40);
    store64be(0x1f83d9abfb41bd6bULL + g, out_hash + 48);
    store64be(0x5be0cd19137e2179ULL + h, out_hash + 56);
}

#endif
