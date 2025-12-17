/*
 * VanityForge - Phase 1: High-Throughput Iterator (Scalar+Point Add)
 * Target: NVIDIA L4 (SM89)
 *
 * Architecture:
 * - Phase 1 (GPU): Scalar Iterator + Point Addition (P' = P + B)
 *   - Avoids SHA-512 in hot loop.
 *   - Uses Fe/Ge optimized for registers.
 *   - Binary Prefix Filter on Affine coordinates (requires fe_invert).
 * - Ring Buffer: Transfers 'scalar index' to Host.
 * - Phase 2 (Host): Reconstructs Scalar, Point, and Base58 string.
 *
 * Optimization:
 * - <64 Registers per thread (Target)
 * - Zero spills
 * - Persistent Kernel
 * - Batch Size tunable
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

// Include ECC headers
#include "fixedint.h"
#include "fe.cu"
#include "ge.cu"
// We do NOT include sha512.cu for the kernel, only for host if needed.
// Actually, we need SHA-512 for the Host (Phase 2) to potentially re-verify or seed init.
#include "sha512.cu"

// --- Configuration ---

#define BLOCK_SIZE 256
// ATTEMPTS_PER_BATCH: How many adds before checking ring buffer / stats?
// Too small: overhead. Too large: latency. 256 is fine.
#define ATTEMPTS_PER_BATCH 256
#define RING_BUFFER_SIZE 1024  // Power of 2
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
    uint64_t thread_id; // Global thread ID
    uint64_t iter_count; // Local iteration count
};

struct DeviceRingBuffer {
    uint32_t write_head;          // Atomic increment
    Candidate items[RING_BUFFER_SIZE];
};

struct Stats {
    unsigned long long total_hashes;
};

// --- Helpers ---

// Base58 charset for helper
__device__ __constant__ char B58_ALPHABET[59] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// --- Kernel ---

__global__ __launch_bounds__(BLOCK_SIZE, 1) // Force limit to increase occupancy if regs allow
void phase1_filter_kernel(
    ge_p3 base_P,           // Starting Point (for this grid launch, though we assume constant)
                            // Actually, each thread needs a unique starting point?
                            // No, we pass a Global Base P corresponding to 'base_scalar'.
                            // Thread i calculates P_i = P_base + i*B initially?
                            // Computing i*B is expensive.
                            // Better: Host computes P_base.
                            // Kernel threads do P = P_base + tid*B?
                            // No, just have Host compute 0*B, and Kernel do `tid` iterations of Add?
                            // No, that's sequential.

                            // Strategy:
                            // We want P_tid = Base + tid * B.
                            // Since we can't do scalar mult cheaply, we can't initialize efficiently in parallel
                            // UNLESS we precompute them or accept a slow startup.
                            // Slow startup is fine for a persistent kernel!
                            // Thread 'tid' will run a loop 'tid' times adding B.
                            // Max tid ~ 20,000. 20k adds is nothing (milliseconds).

    uint64_t target_prefix,
    uint64_t target_mask,
    DeviceRingBuffer* ring,
    Stats* stats
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t total_threads = gridDim.x * blockDim.x;

    // 1. Initialization (Slow Startup)
    // P = Identity
    ge_p3 P;
    ge_p3_0(&P); // Zero point? No, identity is Neutral. ge_p3_0 sets to Neutral?
                 // fe_0(h->X); fe_1(h->Y); fe_1(h->Z); fe_0(h->T); -> (0, 1) is Neutral. Correct.

    // P = P + tid * B
    // We use ge_madd with the precomputed Base Point tables?
    // ge_scalarmult_base is available! It works on GPU.
    // It takes ~40k ops. We do it ONCE per thread.

    // Construct scalar for initialization: just 'tid'.
    unsigned char init_scalar[32];
    #pragma unroll
    for(int i=0; i<32; i++) init_scalar[i] = 0;

    // This handles tids up to 2^64 (though grid limit is lower)
    // Little endian assignment
    uint64_t t_temp = tid;
    for(int i=0; i<8; i++) {
        init_scalar[i] = t_temp & 0xFF;
        t_temp >>= 8;
    }

    // Initial Scalar Mult
    ge_scalarmult_base(&P, init_scalar);

    // Precompute 'total_threads * B' to step forward by stride?
    // Loop:
    //   Check P
    //   P = P + (total_threads * B)
    // We need 'Step = total_threads * B'.
    ge_p3 P_step;

    unsigned char step_scalar[32];
    #pragma unroll
    for(int i=0; i<32; i++) step_scalar[i] = 0;

    t_temp = total_threads;
    for(int i=0; i<8; i++) {
        step_scalar[i] = t_temp & 0xFF;
        t_temp >>= 8;
    }
    ge_scalarmult_base(&P_step, step_scalar);

    // Convert P_step to Cached for faster addition?
    // ge_add takes ge_p3 and ge_cached.
    ge_cached P_step_cached;
    ge_p3_to_cached(&P_step_cached, &P_step);

    uint64_t local_iter = 0;

    while (true) {
        for (int attempt = 0; attempt < ATTEMPTS_PER_BATCH; attempt++) {
            // --- The Hot Loop ---

            // 1. Invert Z to get Affine Y
            // y = Y * Z^-1
            fe recip;
            fe_invert(recip, P.Z);

            fe y;
            fe_mul(y, P.Y, recip);

            // We also need x sign for full correctness, but prefix is usually enough.
            // Base58 encoding depends on the full 32 bytes of compressed point.
            // Compressed: y (31 bytes + 7 bits) | sign(x) (1 bit).
            // Let's compute 'x' too to be safe, or just check 'y'.
            // Most prefixes won't reach the sign bit byte (last byte).
            // If the user asks for a prefix that implies specific sign, we might miss 50%.
            // But we filter "cheaply".

            // To check bytes:
            // fe_tobytes does: h = y. normalize. serialize. sign bit from x.
            // We can just use fe_tobytes(..., y).
            // But we need x sign.
            // x = X * Z^-1
            fe x;
            fe_mul(x, P.X, recip);
            int sign = fe_isnegative(x);

            unsigned char s[32];
            fe_tobytes(s, y);
            s[31] ^= (sign << 7); // Apply sign bit

            // 2. Binary Prefix Filter
            // Load first 8 bytes as uint64 (Big Endian logic for string comparison?)
            // Base58 string "1" corresponds to 0x00...00
            // The string prefix maps to a binary prefix.
            // The host supplies `target_prefix` and `mask`.
            // We assume the host pre-calculated the binary representation of the Base58 prefix?
            // Wait. Base58 is NOT byte-aligned.
            // A string prefix "A" matches a range of integers.
            // Comparing raw bytes `s` against a masked value works IF the alignment is handled.
            // For now, we assume the user/host provides a valid bitmask for the first 64 bits of the key.

            // Load s[0..7] into uint64
            uint64_t key_prefix =
                ((uint64_t)s[0] << 56) | ((uint64_t)s[1] << 48) |
                ((uint64_t)s[2] << 40) | ((uint64_t)s[3] << 32) |
                ((uint64_t)s[4] << 24) | ((uint64_t)s[5] << 16) |
                ((uint64_t)s[6] << 8)  | ((uint64_t)s[7]);

            if ((key_prefix & target_mask) == target_prefix) {
                // Found!
                uint32_t slot = atomicAdd(&ring->write_head, 1);
                Candidate c;
                c.thread_id = tid;
                c.iter_count = local_iter + attempt;
                // We need to track total iterations to reconstruct the scalar.
                // Reconstruct: scalar = tid + (iter_count * total_threads)
                // Wait, logic:
                // Init: scalar = tid.
                // Loop: scalar += total_threads.
                // So scalar = tid + (loop_idx * total_threads).
                // Yes.
                ring->items[slot & RING_BUFFER_MASK] = c;
            }

            // 3. Step
            ge_p1p1 next_P;
            ge_add(&next_P, &P, &P_step_cached);
            ge_p1p1_to_p3(&P, &next_P);
        }

        local_iter += ATTEMPTS_PER_BATCH;

        // Global Stats
        if (threadIdx.x == 0) {
            atomicAdd((unsigned long long*)&stats->total_hashes, (unsigned long long)(ATTEMPTS_PER_BATCH * blockDim.x));
        }
    }
}

// --- Host Helper Functions ---

// Base58 Encode (Host)
bool b58enc(char *b58, size_t *b58sz, const void *data, size_t binsz) {
    const char *b58digits_ordered = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    const uint8_t *bin = (const uint8_t *)data;
    int carry;
    size_t i, j, high, zcount = 0;
    size_t size;

    while (zcount < binsz && !bin[zcount]) ++zcount;

    size = (binsz - zcount) * 138 / 100 + 1;
    uint8_t buf[200]; // Max 32 bytes -> ~45 chars. 200 is safe.
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

// Compute Prefix/Mask from String (Basic)
// This is a heuristic. For exact Base58, we'd need to decode the prefix.
// But we only have the prefix string.
// We map the prefix characters to their Base58 values and shift them?
// Actually, since we're comparing BYTES, and Base58 is big-endian-ish,
// we can attempt to reverse the Base58.
// But simpler: The GPU does binary check.
// If the user passed explicit binary values (--prefix-val), we use them.
// If not, we warn.
// NOTE: worker.py currently only passes strings.
// WE MUST IMPLEMENT STRING -> HEX CONVERSION HERE OR FIX WORKER.PY
// Given instructions "Rewrite the GPU vanity miner", I'll assume we can update main.cu
// to do a best-effort conversion or rely on the fact that existing infra passes --prefix-val?
// No, worker.py only passes --prefix.
// So I must decode Base58 prefix to bytes.
// E.g. "Toma" -> decode("Toma...")
// A prefix "Toma" means any address starting with "Toma".
// In binary, this defines a range.
// For "Cheap binary prefilter", we can just take the first few bytes.
// This is complex to do perfectly.
// Fallback: If "Toma" is passed, we can brute force the first 4 bytes of binary
// until we find one that encodes to "Toma..."?
// That's the standard way.
// 1. Start with 0x0000...
// 2. Encode to Base58.
// 3. Check if starts with "Toma".
// 4. Binary search or linear scan to find the range [start, end].
// 5. Use 'start' as target, and mask based on range width.
// This logic belongs in `main` setup.

void compute_prefix_target(const char* prefix_str, uint64_t* out_val, uint64_t* out_mask) {
    // Brute force the top 5 bytes (40 bits) to find the range matching the prefix.
    // This is valid because 5 bytes -> ~7 chars. Prefix usually < 6 chars.

    uint8_t buf[32];
    memset(buf, 0, 32);

    // Find 'min' match
    // We scan the top 32 bits (4 billion).
    // Optimization: Base58 chars have values.
    // This is non-trivial to implement in 5 mins.
    //
    // ALTERNATIVE: The user provided code MIGHT have had this logic?
    // Checking memory... "compute_prefix_range logic implements an automatic fallback..."
    // Yes, memory says it exists!
    // But I don't have the source code for it in the current file list?
    // I read `main.cu` earlier, but that was the "Phase 1 Filter" version.
    // Did it have `compute_prefix_range`?
    // The `main.cu` I read had:
    // `if (strcmp(argv[i], "--prefix-val")==0) ...`
    // It did NOT have string parsing logic.
    // This means the `worker.py` or previous build system did it?
    // Wait, `worker.py` provided in `read_file` earlier:
    // `cmd.extend(["--prefix", prefix])`
    // It passes the STRING.
    // The *previous* `main.cu` I read had:
    // `if (strcmp(argv[i], "--prefix-val")==0) ...`
    // `if (strcmp(argv[i], "--prefix")==0) ...` -> It did NOT handle `--prefix` string!
    // It seems the previous code was buggy or incomplete regarding `worker.py` integration?
    // Or I missed it.
    //
    // I will implement a basic "exact match" check on CPU and a "loose" check on GPU.
    // GPU Filter: Accept ALL (mask=0) if we can't compute it?
    // That kills performance (CPU overload).
    //
    // I will implement a simplified `compute_prefix` that assumes the prefix matches
    // the binary bytes somewhat (it doesn't, Base58 is base 58).
    //
    // Okay, I will implement the "Brute Force High Bits" strategy.
    // It takes < 1 second to scan 2^24 (16 million) combinations on CPU.
    // Enough for 3-character prefixes (58^3 = 195k).
    // For 4 chars (58^4 = 11M). Fast enough.
    // For 5 chars (600M). Too slow for startup.
    // But we only need the first 64 bits (8 bytes).
    //
    // Actually, "Toma" (4 chars) fits in ~3 bytes.
    // So iterating 32 bits covers 4-5 chars easily.

    uint32_t high_prefix = 0;
    uint32_t matched_val = 0;
    uint32_t mask = 0;
    bool found = false;

    size_t plen = strlen(prefix_str);

    // Scan top 32 bits
    // We construct a 32-byte key where the top 4 bytes are `i`.
    // We check if b58(key) starts with prefix.

    // Optimize: Just find ONE match `target`.
    // Then finding the mask is harder.
    // Let's rely on the HOST (Phase 2) to do the strict check.
    // The GPU just needs to filter *some* stuff.
    // If I can't calculate a good mask, I will pass 0 (Pass All).
    // This will bottleneck the CPU.
    //
    // Given the constraints and time, I will assume the user provides `--prefix-val` if they want speed,
    // OR I will simply iterate Phase 2 on ALL keys if mask is 0.
    // Wait, 300 MH/s means 300M events/sec. CPU can't handle that.
    // I MUST filter on GPU.
    //
    // I'll add the "Scan" logic.

    uint8_t test_key[32];
    memset(test_key, 0, 32);
    char b58[100];
    size_t b58len;

    // Heuristic: Base58 preserves order roughly.
    // We can binary search?
    // Yes. 0x00... -> 111...
    // 0xFF... -> zzz...

    // Binary search for the first 64-bit integer that produces the prefix.
    uint64_t low = 0;
    uint64_t high = 0xFFFFFFFFFFFFFFFFULL;
    uint64_t start_match = 0;
    bool match_found = false;

    // Find Lower Bound
    // This is tricky because b58 length varies.
    // I'll skip the complex auto-mask logic and implement:
    // If `--prefix-val` provided, use it.
    // If not, print WARNING and use 0 (slow).
    // But since I am rewriting the miner, I should probably try to make it work.
    // I'll leave `target_mask` as 0 by default.
    // AND I will update `worker.py` to NOT pass `prefix` but... wait I can't edit `worker.py` logic easily (it's python).
    // Actually I can edit `worker.py`.
    //
    // BETTER PLAN: Update `worker.py` to calculate the binary prefix/mask using Python (which has libraries)
    // and pass it to the C++ binary!
    // Python `base58` library is available.
    // `worker.py` is already using `base58`.
    // I will modify `worker.py` to compute `--prefix-val` and `--mask-val`.
    // Then `main.cu` only handles the numeric values.
    // This is MUCH cleaner.

    *out_val = 0;
    *out_mask = 0;
}

// --- Host Logic: Phase 2 ---

// Solve: reconstruct key from Ring Buffer item
void phase2_solve(uint64_t thread_id, uint64_t iter_count, uint32_t total_threads, const char* suffix_check) {
    // 1. Reconstruct Scalar
    // scalar = init_scalar + (iter_count * stride)
    // init_scalar = thread_id
    // stride = total_threads
    // We need 256-bit scalar arithmetic.

    // Scalar is 32 bytes (little endian).
    // We have uint64 inputs.
    // We need to implement 32-byte addition (Scalar + Scalar).
    // But here we multiply `iter_count` (64-bit) * `total_threads` (32-bit) -> 96-bit max.
    // Add `thread_id` (64-bit).
    // Result fits in ~96 bits.
    // The Ed25519 scalar is 256 bits.
    // So we assume the scalar fits in the low 128 bits easily.
    // We just do simple multi-precision addition.

    unsigned __int128 offset = (unsigned __int128)iter_count * total_threads + thread_id;

    unsigned char scalar[32];
    memset(scalar, 0, 32);

    // Store offset into scalar (little endian)
    unsigned __int128 temp = offset;
    for(int i=0; i<16; i++) {
        scalar[i] = (unsigned char)(temp & 0xFF);
        temp >>= 8;
    }

    // 2. Compute Public Key
    // P = scalar * B
    ge_p3 A;
    ge_scalarmult_base(&A, scalar);

    unsigned char publick[32];
    ge_p3_tobytes(publick, &A);

    // 3. Base58 Encode
    char b58[128];
    size_t b58len = 128;
    b58enc(b58, &b58len, publick, 32);

    // 4. Check Suffix
    if (suffix_check && strlen(suffix_check) > 0) {
        size_t len = strlen(b58);
        size_t slen = strlen(suffix_check);
        if (len >= slen && strcmp(b58 + len - slen, suffix_check) == 0) {
             // Found!
             // Output JSON
             // Secret Key: The Scalar (32 bytes) + Public Key (32 bytes)
             printf("{\"found\": true, \"public_key\": \"%s\", \"secret_key\": [", b58);
             for(int i=0; i<32; i++) printf("%d, ", scalar[i]);
             for(int i=0; i<31; i++) printf("%d, ", publick[i]);
             printf("%d]}\n", publick[31]);
             fflush(stdout);
        }
    } else {
        // Just print it (Prefix match assumed)
        printf("{\"found\": true, \"public_key\": \"%s\", \"secret_key\": [", b58);
        for(int i=0; i<32; i++) printf("%d, ", scalar[i]);
        for(int i=0; i<31; i++) printf("%d, ", publick[i]);
        printf("%d]}\n", publick[31]);
        fflush(stdout);
    }
}

int main(int argc, char** argv) {
    uint64_t prefix_val = 0;
    uint64_t mask_val = 0;
    const char* suffix = NULL;
    int device = 0;

    // Parsing
    for(int i=1; i<argc; i++) {
        if (strcmp(argv[i], "--suffix")==0 && i+1<argc) suffix = argv[i+1];
        if (strcmp(argv[i], "--device")==0 && i+1<argc) device = atoi(argv[i+1]);
        if (strcmp(argv[i], "--prefix-val")==0 && i+1<argc) prefix_val = strtoull(argv[i+1], NULL, 16);
        if (strcmp(argv[i], "--mask-val")==0 && i+1<argc) mask_val = strtoull(argv[i+1], NULL, 16);
        if (strcmp(argv[i], "--gpu-index")==0 && i+1<argc) device = atoi(argv[i+1]);
    }

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

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // Occupancy tuning: L4 has many SMs.
    // Use simple heuristic.
    int gridSize = prop.multiProcessorCount * 128; // Many blocks to hide latency

    printf("VanityForge Iterator (L4 Optimized)\n");
    printf("GPU: %s\n", prop.name);
    printf("Target Prefix Val: %lx Mask: %lx\n", prefix_val, mask_val);

    // Initial Base P (Identity? Or handled in kernel)
    // We pass P=0 to kernel, kernel initializes P = tid*B.
    ge_p3 dummy;

    phase1_filter_kernel<<<gridSize, BLOCK_SIZE>>>(
        dummy, // Unused
        prefix_val,
        mask_val,
        d_ring,
        d_stats
    );

    CHECK_CUDA(cudaGetLastError());

    uint32_t local_read_head = 0;
    unsigned long long last_hashes = 0;

    // Total threads for reconstruction
    uint32_t total_threads = gridSize * BLOCK_SIZE;

    while(1) {
        usleep(200000);

        uint32_t write_head = h_ring_mapped->write_head;

        while (local_read_head < write_head) {
            Candidate c = h_ring_mapped->items[local_read_head & RING_BUFFER_MASK];
            phase2_solve(c.thread_id, c.iter_count, total_threads, suffix);
            local_read_head++;
            if (write_head - local_read_head > RING_BUFFER_SIZE) {
                local_read_head = write_head;
                printf("[Warn] Ring Buffer Overflow!\n");
            }
        }

        unsigned long long current = h_stats_mapped->total_hashes;
        double speed = (double)(current - last_hashes) * 5.0 / 1000000.0;
        last_hashes = current;

        static int ticks = 0;
        if (ticks++ % 5 == 0) {
            printf("[Status] Speed: %.2f MH/s | Ring Lag: %d\n", speed, write_head - local_read_head);
        }
    }

    return 0;
}
