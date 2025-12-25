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
#include <fcntl.h>
#include <random>
#include <chrono>
#include <vector>
#include <string>

// Include ECC headers
#include "fixedint.h"
#include "fe.cu"
#include "ge.h"           // Define ge_precomp struct before using it in precomp_data.h
#include "precomp_data.h" // Include BEFORE ge.cu so 'base' is visible
#include "ge.cu"
// We do NOT include sha512.cu for the kernel, only for host if needed.
// Actually, we need SHA-512 for the Host (Phase 2) to potentially re-verify or seed init.
#include "sha512.cu"

// --- Macros ---

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (Line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// --- Precomputation Management ---

void generate_tables() {
    FILE* f = fopen("precomp_tables.bin", "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open precomp_tables.bin for writing.\n");
        exit(1);
    }
    // base is defined in precomp_data.h
    size_t written = fwrite(base, sizeof(ge_precomp), 32 * 8, f);
    if (written != 32 * 8) {
        fprintf(stderr, "Error: Failed to write all table data.\n");
        exit(1);
    }
    fclose(f);
    printf("Precomputed tables generated: precomp_tables.bin\n");
}

void load_tables() {
    FILE* f = fopen("precomp_tables.bin", "rb");
    if (!f) {
        // Fallback: Generate if missing (Auto-healing)
        printf("Tables not found. Generating...\n");
        generate_tables();
        f = fopen("precomp_tables.bin", "rb");
        if (!f) {
            fprintf(stderr, "Error: Could not open precomp_tables.bin for reading.\n");
            exit(1);
        }
    }

    // Allocate temp buffer
    ge_precomp* host_base = (ge_precomp*)malloc(sizeof(ge_precomp) * 32 * 8);
    if (!host_base) {
        fprintf(stderr, "Error: Malloc failed for host_base.\n");
        exit(1);
    }

    size_t read_count = fread(host_base, sizeof(ge_precomp), 32 * 8, f);
    fclose(f);

    if (read_count != 32 * 8) {
        fprintf(stderr, "Error: precomp_tables.bin is corrupted (Read %zu items).\n", read_count);
        // Try to regenerate once?
        generate_tables();
        // Recurse once? No, simpler to just exit or reload.
        // For robustness, let's just exit.
        exit(1);
    }

    // Copy to Constant Memory
    // c_base is defined in ge.cu
    CHECK_CUDA(cudaMemcpyToSymbol(c_base, host_base, sizeof(ge_precomp) * 32 * 8));
    free(host_base);
    printf("Precomputed tables loaded to Constant Memory.\n");
}

// --- Configuration ---

#define BLOCK_SIZE 256
// ATTEMPTS_PER_BATCH: How many adds before checking ring buffer / stats?
// Too small: overhead. Too large: latency. 256 is fine.
#define ATTEMPTS_PER_BATCH 4096
#define BATCH_SIZE ATTEMPTS_PER_BATCH // Alias for clarity with user instructions
#define RING_BUFFER_SIZE 1024  // Power of 2
#define RING_BUFFER_MASK (RING_BUFFER_SIZE - 1)

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
    ge_p3 base_P,           // Starting Point (Unused, we init from scalar)
    uint64_t min_limit,
    uint64_t max_limit,
    uint64_t random_offset, // Randomized Offset
    DeviceRingBuffer* ring,
    Stats* stats
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t total_threads = gridDim.x * blockDim.x;

    // 1. Initialization (Slow Startup)
    // P = Identity
    ge_p3 P;
    ge_p3_0(&P);

    // Calculate Start Index with Randomized Offset
    // Formula: start_index = random_offset + tid
    uint64_t start_index = random_offset + (uint64_t)tid;

    // Construct scalar for initialization
    unsigned char init_scalar[32];
    #pragma unroll
    for(int i=0; i<32; i++) init_scalar[i] = 0;

    // This handles tids up to 2^64 (though grid limit is lower)
    // Little endian assignment
    uint64_t t_temp = start_index;
    for(int i=0; i<8; i++) {
        init_scalar[i] = t_temp & 0xFF;
        t_temp >>= 8;
    }

    // Initial Scalar Mult
    // P = start_index * B
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

    // Convert P_step to Cached for faster addition
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
            fe x;
            fe_mul(x, P.X, recip);
            int sign = fe_isnegative(x);

            unsigned char s[32];
            fe_tobytes(s, y);
            s[31] ^= (sign << 7); // Apply sign bit

            // 2. Binary Prefix Filter (Top-64 Bit Range Check)
            // We need to construct the 64-bit integer Big-Endian style from Little Endian bytes.
            // s[31] is the most significant byte of the Y coordinate.
            uint64_t top64 =
                ((uint64_t)s[31] << 56) | ((uint64_t)s[30] << 48) |
                ((uint64_t)s[29] << 40) | ((uint64_t)s[28] << 32) |
                ((uint64_t)s[27] << 24) | ((uint64_t)s[26] << 16) |
                ((uint64_t)s[25] << 8)  | ((uint64_t)s[24]);

            // Range Check
            if (top64 >= min_limit && top64 <= max_limit) {
                // Found!
                uint32_t slot = atomicAdd(&ring->write_head, 1);
                Candidate c;
                c.thread_id = tid;
                c.iter_count = local_iter + attempt;
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

// Host constants and helpers for prefix calculation
const char* B58_DIGITS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

int b58_index_host(char c) {
    const char *p = strchr(B58_DIGITS, c);
    if (p) return p - B58_DIGITS;
    return -1;
}

typedef unsigned char uint8_t;

void mul58(uint8_t *n) { // n is 32 bytes LE
    int carry = 0;
    for (int i = 0; i < 32; i++) {
        int val = n[i] * 58 + carry;
        n[i] = val & 0xFF;
        carry = val >> 8;
    }
}

void add_val(uint8_t *n, int v) { // n is 32 bytes LE
    int carry = v;
    for (int i = 0; i < 32 && carry; i++) {
        int val = n[i] + carry;
        n[i] = val & 0xFF;
        carry = val >> 8;
    }
}

void sub_val(uint8_t *n, int v) { // n is 32 bytes LE
    int borrow = v;
    for (int i = 0; i < 32 && borrow; i++) {
        int val = n[i] - borrow;
        if (val < 0) {
            val += 256;
            borrow = 1;
        } else {
            borrow = 0;
        }
        n[i] = val;
    }
}

uint64_t extract_top64(const uint8_t *n) {
    // n is LE, so MSB is at 31
    return ((uint64_t)n[31] << 56) | ((uint64_t)n[30] << 48) |
           ((uint64_t)n[29] << 40) | ((uint64_t)n[28] << 32) |
           ((uint64_t)n[27] << 24) | ((uint64_t)n[26] << 16) |
           ((uint64_t)n[25] << 8)  | ((uint64_t)n[24]);
}

void compute_prefix_bounds(const char* prefix, uint64_t* min_out, uint64_t* max_out) {
    uint8_t min_int[32] = {0};
    uint8_t max_int[32] = {0};

    // Decode P
    for (int i = 0; prefix[i]; i++) {
        mul58(min_int);
        add_val(min_int, b58_index_host(prefix[i]));
    }

    // Copy to max_int
    memcpy(max_int, min_int, 32);

    // For max_int, we start with P+1
    add_val(max_int, 1);

    // Scale up
    int len = strlen(prefix);
    int target_len = 44;
    int shifts = target_len - len;
    if (shifts < 0) shifts = 0;

    for (int i = 0; i < shifts; i++) {
        mul58(min_int);
        mul58(max_int);
    }

    // max_int = (P+1)*shift - 1
    sub_val(max_int, 1);

    *min_out = extract_top64(min_int);
    *max_out = extract_top64(max_int);

    // Safety widening
    *min_out -= 4;
    *max_out += 4;
}

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
void compute_prefix_target(const char* prefix_str, uint64_t* out_val, uint64_t* out_mask) {
    // Stub: This logic is complex and best handled by Python wrapper passing --prefix-val
    // For now, we assume explicit values or accept 0 (all pass).
    *out_val = 0;
    *out_mask = 0;
}

// --- Host Logic: Phase 2 ---

// Solve: reconstruct key from Ring Buffer item
void phase2_solve(uint64_t thread_id, uint64_t iter_count, uint32_t total_threads, uint64_t random_offset, const char* suffix_check, const char* prefix_check) {
    // 1. Reconstruct Scalar
    // Old: scalar = tid + (iter_count * stride)
    // New: scalar = random_offset + tid + (iter_count * total_threads)

    // Use __int128 to prevent overflow before conversion to 32-byte array
    unsigned __int128 offset = (unsigned __int128)random_offset +
                               ((unsigned __int128)thread_id) +
                               ((unsigned __int128)iter_count * total_threads);

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

    // 4. Exact Verification
    if (prefix_check && strlen(prefix_check) > 0) {
        if (strncmp(b58, prefix_check, strlen(prefix_check)) == 0) {
             printf("KEY FOUND\n");
             printf("{\"found\": true, \"public_key\": \"%s\", \"secret_key\": [", b58);
             for(int i=0; i<32; i++) printf("%d, ", scalar[i]);
             for(int i=0; i<31; i++) printf("%d, ", publick[i]);
             printf("%d]}\n", publick[31]);
             fflush(stdout);
             exit(0);
        } else {
             printf("FALSE POSITIVE (Filter)\n");
             return;
        }
    }

    if (suffix_check && strlen(suffix_check) > 0) {
        size_t len = strlen(b58);
        size_t slen = strlen(suffix_check);
        if (len < slen || strcmp(b58 + len - slen, suffix_check) != 0) {
             return;
        }
    }

    // KEY FOUND (Suffix only case, or if no prefix provided)
    printf("{\"found\": true, \"public_key\": \"%s\", \"secret_key\": [", b58);
    for(int i=0; i<32; i++) printf("%d, ", scalar[i]);
    for(int i=0; i<31; i++) printf("%d, ", publick[i]);
    printf("%d]}\n", publick[31]);
    fflush(stdout);
}

int main(int argc, char** argv) {
    uint64_t min_limit = 0;
    uint64_t max_limit = 0xFFFFFFFFFFFFFFFF;
    const char* suffix = NULL;
    const char* prefix_str = NULL;
    int device = 0;

    // Parsing
    for(int i=1; i<argc; i++) {
        if (strcmp(argv[i], "--suffix")==0 && i+1<argc) suffix = argv[i+1];
        if (strcmp(argv[i], "--prefix-str")==0 && i+1<argc) prefix_str = argv[i+1];
        if (strcmp(argv[i], "--prefix")==0 && i+1<argc) prefix_str = argv[i+1];
        if (strcmp(argv[i], "--device")==0 && i+1<argc) device = atoi(argv[i+1]);
        if (strcmp(argv[i], "--min-limit")==0 && i+1<argc) min_limit = strtoull(argv[i+1], NULL, 10);
        if (strcmp(argv[i], "--max-limit")==0 && i+1<argc) max_limit = strtoull(argv[i+1], NULL, 10);
        if (strcmp(argv[i], "--gpu-index")==0 && i+1<argc) device = atoi(argv[i+1]);
        if (strcmp(argv[i], "--generate-tables")==0) {
            generate_tables();
            return 0;
        }
    }

    if (prefix_str) {
        compute_prefix_bounds(prefix_str, &min_limit, &max_limit);
        printf("Computed bounds for prefix '%s': %llu - %llu\n", prefix_str, (unsigned long long)min_limit, (unsigned long long)max_limit);
    }

    cudaSetDevice(device);

    // Load Tables (Fast Path)
    load_tables();

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

    // Generate Randomized Offset
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    uint64_t random_offset = dis(gen);

    printf("VanityForge Iterator (L4 Optimized)\n");
    printf("GPU: %s\n", prop.name);
    printf("Range: %llu - %llu\n", (unsigned long long)min_limit, (unsigned long long)max_limit);
    printf("Random Offset: %lu\n", random_offset);

    // Initial Base P (Identity? Or handled in kernel)
    ge_p3 dummy;

    phase1_filter_kernel<<<gridSize, BLOCK_SIZE>>>(
        dummy, // Unused
        min_limit,
        max_limit,
        random_offset,
        d_ring,
        d_stats
    );

    CHECK_CUDA(cudaGetLastError());

    uint32_t local_read_head = 0;
    unsigned long long last_hashes = 0;
    auto last_time = std::chrono::high_resolution_clock::now();

    // Total threads for reconstruction
    uint32_t total_threads = gridSize * BLOCK_SIZE;

    while(1) {
        usleep(50000);

        uint32_t write_head = h_ring_mapped->write_head;

        while (local_read_head < write_head) {
            Candidate c = h_ring_mapped->items[local_read_head & RING_BUFFER_MASK];
            phase2_solve(c.thread_id, c.iter_count, total_threads, random_offset, suffix, prefix_str);
            local_read_head++;
            if (write_head - local_read_head > RING_BUFFER_SIZE) {
                local_read_head = write_head;
                printf("[Warn] Ring Buffer Overflow!\n");
            }
        }

        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - last_time;

        if (elapsed.count() >= 1.0) {
            unsigned long long current = h_stats_mapped->total_hashes;
            double speed = (double)(current - last_hashes) / elapsed.count() / 1000000.0;
            printf("[Status] Speed: %.2f MH/s | Ring Lag: %d\n", speed, write_head - local_read_head);

            last_hashes = current;
            last_time = current_time;
        }
    }

    return 0;
}
