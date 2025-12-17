/*
 * VanityForge - Phase 1: High-Throughput SHA-512 Filter
 * Target: NVIDIA L4 (SM89)
 *
 * Architecture:
 * - Phase 1 (GPU): Pure SHA-512 generation + Binary Prefix Filter.
 * - Ring Buffer: Transfers candidates to Host/Phase 2.
 * - Phase 2 (Host): Consumes candidates, performs Ed25519 + Base58 check.
 *
 * Optimization:
 * - <64 Registers per thread
 * - Zero spills
 * - Persistent Kernel
 * - Rolling SHA-512 Schedule
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

// Include ECC headers for Phase 2 (Host side verification)
#include "fixedint.h"
#include "fe.cu"
#include "ge.cu"
#include "sc.cu"
#include "sha512.cu"

// --- Configuration ---

#define BLOCK_SIZE 256
#define ATTEMPTS_PER_BATCH 256 // Hashes per thread before checking buffer/stats
#define RING_BUFFER_SIZE 4096  // Must be power of 2
#define RING_BUFFER_MASK (RING_BUFFER_SIZE - 1)

// --- Macros ---

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (Line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// --- Data Structures ---

struct Candidate {
    uint64_t seed_idx; // Unique ID to reconstruct seed
};

struct DeviceRingBuffer {
    uint32_t write_head;          // Atomic increment
    Candidate items[RING_BUFFER_SIZE];
};

struct Stats {
    unsigned long long total_hashes;
};

// --- SHA-512 Constants (Kernel) ---
// Renamed to avoid conflict with sha512.cu 'K'
__device__ const uint64_t K_SHA512[80] = {
    UINT64_C(0x428a2f98d728ae22), UINT64_C(0x7137449123ef65cd),
    UINT64_C(0xb5c0fbcfec4d3b2f), UINT64_C(0xe9b5dba58189dbbc),
    UINT64_C(0x3956c25bf348b538), UINT64_C(0x59f111f1b605d019),
    UINT64_C(0x923f82a4af194f9b), UINT64_C(0xab1c5ed5da6d8118),
    UINT64_C(0xd807aa98a3030242), UINT64_C(0x12835b0145706fbe),
    UINT64_C(0x243185be4ee4b28c), UINT64_C(0x550c7dc3d5ffb4e2),
    UINT64_C(0x72be5d74f27b896f), UINT64_C(0x80deb1fe3b1696b1),
    UINT64_C(0x9bdc06a725c71235), UINT64_C(0xc19bf174cf692694),
    UINT64_C(0xe49b69c19ef14ad2), UINT64_C(0xefbe4786384f25e3),
    UINT64_C(0x0fc19dc68b8cd5b5), UINT64_C(0x240ca1cc77ac9c65),
    UINT64_C(0x2de92c6f592b0275), UINT64_C(0x4a7484aa6ea6e483),
    UINT64_C(0x5cb0a9dcbd41fbd4), UINT64_C(0x76f988da831153b5),
    UINT64_C(0x983e5152ee66dfab), UINT64_C(0xa831c66d2db43210),
    UINT64_C(0xb00327c898fb213f), UINT64_C(0xbf597fc7beef0ee4),
    UINT64_C(0xc6e00bf33da88fc2), UINT64_C(0xd5a79147930aa725),
    UINT64_C(0x06ca6351e003826f), UINT64_C(0x142929670a0e6e70),
    UINT64_C(0x27b70a8546d22ffc), UINT64_C(0x2e1b21385c26c926),
    UINT64_C(0x4d2c6dfc5ac42aed), UINT64_C(0x53380d139d95b3df),
    UINT64_C(0x650a73548baf63de), UINT64_C(0x766a0abb3c77b2a8),
    UINT64_C(0x81c2c92e47edaee6), UINT64_C(0x92722c851482353b),
    UINT64_C(0xa2bfe8a14cf10364), UINT64_C(0xa81a664bbc423001),
    UINT64_C(0xc24b8b70d0f89791), UINT64_C(0xc76c51a30654be30),
    UINT64_C(0xd192e819d6ef5218), UINT64_C(0xd69906245565a910),
    UINT64_C(0xf40e35855771202a), UINT64_C(0x106aa07032bbd1b8),
    UINT64_C(0x19a4c116b8d2d0c8), UINT64_C(0x1e376c085141ab53),
    UINT64_C(0x2748774cdf8eeb99), UINT64_C(0x34b0bcb5e19b48a8),
    UINT64_C(0x391c0cb3c5c95a63), UINT64_C(0x4ed8aa4ae3418acb),
    UINT64_C(0x5b9cca4f7763e373), UINT64_C(0x682e6ff3d6b2b8a3),
    UINT64_C(0x748f82ee5defb2fc), UINT64_C(0x78a5636f43172f60),
    UINT64_C(0x84c87814a1f0ab72), UINT64_C(0x8cc702081a6439ec),
    UINT64_C(0x90befffa23631e28), UINT64_C(0xa4506cebde82bde9),
    UINT64_C(0xbef9a3f7b2c67915), UINT64_C(0xc67178f2e372532b),
    UINT64_C(0xca273eceea26619c), UINT64_C(0xd186b8c721c0c207),
    UINT64_C(0xeada7dd6cde0eb1e), UINT64_C(0xf57d4f7fee6ed178),
    UINT64_C(0x06f067aa72176fba), UINT64_C(0x0a637dc5a2c898a6),
    UINT64_C(0x113f9804bef90dae), UINT64_C(0x1b710b35131c471b),
    UINT64_C(0x28db77f523047d84), UINT64_C(0x32caab7b40c72493),
    UINT64_C(0x3c9ebe0a15c9bebc), UINT64_C(0x431d67c49c100d4c),
    UINT64_C(0x4cc5d4becb3e42b6), UINT64_C(0x597f299cfc657e2a),
    UINT64_C(0x5fcb6fab3ad6faec), UINT64_C(0x6c44198c4a475817)
};

#define ROR64c(x, y) \
    ( ((((x)&UINT64_C(0xFFFFFFFFFFFFFFFF))>>((uint64_t)(y)&UINT64_C(63))) | \
      ((x)<<((uint64_t)(64-((y)&UINT64_C(63)))))) & UINT64_C(0xFFFFFFFFFFFFFFFF))

#define Ch(x,y,z)       (z ^ (x & (y ^ z)))
#define Maj(x,y,z)      (((x | y) & z) | (x & y))
#define S(x, n)         ROR64c(x, n)
#define R(x, n)         (((x) &UINT64_C(0xFFFFFFFFFFFFFFFF))>>((uint64_t)n))
#define Sigma0(x)       (S(x, 28) ^ S(x, 34) ^ S(x, 39))
#define Sigma1(x)       (S(x, 14) ^ S(x, 18) ^ S(x, 41))
#define Gamma0(x)       (S(x, 1) ^ S(x, 8) ^ R(x, 7))
#define Gamma1(x)       (S(x, 19) ^ S(x, 61) ^ R(x, 6))

__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t lo = (uint32_t)(x & 0xFFFFFFFF);
    hi = __byte_perm(hi, 0, 0x0123);
    lo = __byte_perm(lo, 0, 0x0123);
    return ((uint64_t)lo << 32) | hi;
}

// --- Phase 1 Kernel ---

__global__ __launch_bounds__(BLOCK_SIZE, 2)
void phase1_filter_kernel(
    uint64_t base_seed,
    uint64_t target_prefix,
    uint64_t target_mask,
    DeviceRingBuffer* ring,
    Stats* stats
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t total_threads = gridDim.x * blockDim.x;

    uint64_t current_idx = base_seed + tid;

    // IV
    const uint64_t IV[8] = {
        UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
        UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
        UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
        UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)
    };

    while (true) {
        for (int attempt = 0; attempt < ATTEMPTS_PER_BATCH; attempt++) {

            // State
            uint64_t S[8];
            #pragma unroll
            for(int i=0; i<8; i++) S[i] = IV[i];

            // Schedule
            uint64_t W[16];
            #pragma unroll
            for(int i=0; i<16; i++) W[i] = 0;

            // Set Seed (32 bytes)
            W[0] = current_idx;
            W[4] = UINT64_C(0x8000000000000000);
            W[15] = 256;

            uint64_t t0, t1;

            // SHA Rounds
            // Loop 0..79
            for (int i = 0; i < 80; i++) {
                // Update W
                uint64_t Wi;
                if (i < 16) {
                    Wi = W[i];
                } else {
                    // W[i] = Gamma1(W[i-2]) + W[i-7] + Gamma0(W[i-15]) + W[i-16]
                    Wi = Gamma1(W[(i-2)&15]) + W[(i-7)&15] + Gamma0(W[(i-15)&15]) + W[(i)&15];
                    W[(i)&15] = Wi;
                }

                // Round function
                // Uses K_SHA512 to avoid conflict with host sha512.cu
                t0 = S[7] + Sigma1(S[4]) + Ch(S[4], S[5], S[6]) + K_SHA512[i] + Wi;
                t1 = Sigma0(S[0]) + Maj(S[0], S[1], S[2]);

                // Shift
                S[7] = S[6];
                S[6] = S[5];
                S[5] = S[4];
                S[4] = S[3] + t0;
                S[3] = S[2];
                S[2] = S[1];
                S[1] = S[0];
                S[0] = t0 + t1;
            }

            // --- Step C: Filter ---
            uint64_t digest0 = S[0] + IV[0];

            if ((digest0 & target_mask) == target_prefix) {
                // Found Candidate!
                uint32_t slot = atomicAdd(&ring->write_head, 1);
                ring->items[slot & RING_BUFFER_MASK].seed_idx = current_idx;
            }

            // Increment Seed
            current_idx += total_threads;
        }

        // Global Stats
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd((unsigned long long*)&stats->total_hashes, (unsigned long long)(ATTEMPTS_PER_BATCH * blockDim.x));
        }
        __syncthreads();
    }
}

// --- Host Logic: Phase 2 ---

// Helper for Base58
bool b58enc_host(char *b58, size_t *b58sz, const uint8_t *data, size_t binsz) {
    const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    const uint8_t *bin = data;
    int carry;
    size_t i, j, high, zcount = 0;
    size_t size;

    while (zcount < binsz && !bin[zcount]) ++zcount;

    size = (binsz - zcount) * 138 / 100 + 1;
    uint8_t buf[256];
    memset(buf, 0, size);

    for (i = zcount, high = size - 1; i < binsz; ++i, high = j) {
        for (carry = bin[i], j = size - 1; (j > high) || carry; --j) {
            carry += 256 * buf[j];
            buf[j] = carry % 58;
            carry /= 58;
            if (!j) break;
        }
    }

    for (j = 0; j < size && !buf[j]; ++j);

    if (*b58sz <= zcount + size - j) {
        *b58sz = zcount + size - j + 1;
        return false;
    }

    if (zcount) memset(b58, '1', zcount);
    for (i = zcount; j < size; ++i, ++j) b58[i] = b58digits_ordered[buf[j]];
    b58[i] = '\0';
    *b58sz = i + 1;
    return true;
}

// Host SHA-512 Helper
void sha512_host(const uint8_t* msg, size_t len, uint8_t* out) {
    // We use the existing SHA-512 implementation from sha512.cu or openssl if linked.
    // Assuming sha512.cu defines 'sha512' function
    sha512(msg, len, out);
}

// Host Phase 2 Solver
void phase2_solve(uint64_t seed_idx, const char* suffix_check) {
    // 1. Reconstruct Seed (32 bytes)
    uint8_t seed[32];
    memset(seed, 0, 32);
    // seed[0..7] = seed_idx (Big Endian as per kernel)

    for(int i=0; i<8; i++) {
        seed[i] = (seed_idx >> (56 - 8*i)) & 0xFF;
    }

    // 2. SHA-512 to get Scalar
    uint8_t digest[64];
    sha512_host(seed, 32, digest);

    // 3. Ed25519 Scalar Mult
    unsigned char privatek[32];
    memcpy(privatek, digest, 32);

    // Clamp
    privatek[0] &= 248;
    privatek[31] &= 63;
    privatek[31] |= 64;

    ge_p3 A;
    ge_scalarmult_base(&A, privatek);

    unsigned char publick[32];
    ge_p3_tobytes(publick, &A);

    // 4. Base58 Check
    char b58[128];
    size_t b58len = 128;
    b58enc_host(b58, &b58len, publick, 32);

    // 5. Verification (Suffix)
    if (suffix_check && strlen(suffix_check) > 0) {
        size_t len = strlen(b58);
        size_t slen = strlen(suffix_check);
        if (len >= slen && strcmp(b58 + len - slen, suffix_check) == 0) {
             printf("{\"found\": true, \"public_key\": \"%s\", \"seed_idx\": %lu}\n", b58, seed_idx);
        }
    } else {
        // Just print it
        printf("{\"found\": true, \"public_key\": \"%s\", \"seed_idx\": %lu}\n", b58, seed_idx);
    }
}

int main(int argc, char** argv) {
    // 1. Arguments
    uint64_t prefix = 0;
    uint64_t mask = 0;
    const char* suffix = NULL;
    int device = 0;

    // Parsing
    for(int i=1; i<argc; i++) {
        if (strcmp(argv[i], "--suffix")==0 && i+1<argc) suffix = argv[i+1];
        if (strcmp(argv[i], "--device")==0 && i+1<argc) device = atoi(argv[i+1]);
        if (strcmp(argv[i], "--prefix-val")==0 && i+1<argc) prefix = strtoull(argv[i+1], NULL, 16);
        if (strcmp(argv[i], "--mask-val")==0 && i+1<argc) mask = strtoull(argv[i+1], NULL, 16);
    }

    if (mask == 0) {
        printf("[Warn] Mask is 0. Phase 1 will pass ALL candidates (Performance hit!). Use --mask-val.\n");
    }

    // 2. Setup
    cudaSetDevice(device);

    DeviceRingBuffer* d_ring;
    Stats* d_stats;
    DeviceRingBuffer* h_ring_mapped;
    Stats* h_stats_mapped;

    cudaSetDeviceFlags(cudaDeviceMapHost);

    cudaHostAlloc(&h_ring_mapped, sizeof(DeviceRingBuffer), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_ring, h_ring_mapped, 0);

    cudaHostAlloc(&h_stats_mapped, sizeof(Stats), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_stats, h_stats_mapped, 0);

    memset(h_ring_mapped, 0, sizeof(DeviceRingBuffer));
    memset(h_stats_mapped, 0, sizeof(Stats));

    // 3. Launch Phase 1
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int gridSize = prop.multiProcessorCount * 32;

    printf("VanityForge Phase 1+2 (Filter + Solver)\n");
    printf("GPU: %s\n", prop.name);

    phase1_filter_kernel<<<gridSize, BLOCK_SIZE>>>(
        0ULL,
        prefix,
        mask,
        d_ring,
        d_stats
    );

    CHECK_CUDA(cudaGetLastError());

    // 4. Phase 2 Host Loop
    uint32_t local_read_head = 0;
    unsigned long long last_hashes = 0;

    while(1) {
        usleep(200000); // 0.2s check interval for responsiveness

        // 4.1 Process Ring Buffer
        uint32_t write_head = h_ring_mapped->write_head;

        while (local_read_head < write_head) {
            Candidate c = h_ring_mapped->items[local_read_head & RING_BUFFER_MASK];
            phase2_solve(c.seed_idx, suffix);
            local_read_head++;
            // Don't fall too far behind, but usually Phase 1 filters aggressively
            if (write_head - local_read_head > RING_BUFFER_SIZE) {
                // Buffer overflowed, skip to head
                local_read_head = write_head;
                printf("[Warn] Ring Buffer Overflow! Skipping...\n");
            }
        }

        // 4.2 Stats
        unsigned long long current = h_stats_mapped->total_hashes;
        double speed = (double)(current - last_hashes) * 5.0 / 1000000.0; // 5.0 because 0.2s interval
        last_hashes = current;

        // Only print stats periodically to avoid spam
        static int ticks = 0;
        if (ticks++ % 5 == 0) {
            printf("[Status] Speed: %.2f MH/s | Ring Lag: %d\n", speed, write_head - local_read_head);
        }
    }

    return 0;
}
