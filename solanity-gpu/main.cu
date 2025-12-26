/*
 * VanityForge - Phase 1: High-Throughput Seed Grinder (SHA512 + BaseMult)
 * Target: NVIDIA L4 (SM89)
 *
 * Architecture:
 * - Phase 1 (GPU): Seed Iterator -> SHA512 -> Clamp -> Point Mult
 *   - CORRECTS Ed25519 Invariant (Seed-based keys).
 *   - Slower than Scalar Iterator, but cryptographically standard.
 *   - Base58 Prefix Filter (Probabilistic/Safe).
 * - Ring Buffer: Transfers 'seed index' to Host.
 * - Phase 2 (Host): Reconstructs Seed, Keypair, and Base58 string.
 *
 * Optimization:
 * - <64 Registers per thread (Target)
 * - Zero spills
 * - Persistent Kernel
 * - Batch Size reduced for heavy kernel
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
        printf("Tables not found. Generating...\n");
        generate_tables();
        f = fopen("precomp_tables.bin", "rb");
        if (!f) {
            fprintf(stderr, "Error: Could not open precomp_tables.bin for reading.\n");
            exit(1);
        }
    }

    ge_precomp* host_base = (ge_precomp*)malloc(sizeof(ge_precomp) * 32 * 8);
    if (!host_base) {
        fprintf(stderr, "Error: Malloc failed for host_base.\n");
        exit(1);
    }

    size_t read_count = fread(host_base, sizeof(ge_precomp), 32 * 8, f);
    fclose(f);

    if (read_count != 32 * 8) {
        fprintf(stderr, "Error: precomp_tables.bin is corrupted (Read %zu items).\n", read_count);
        exit(1);
    }

    CHECK_CUDA(cudaMemcpyToSymbol(c_base, host_base, sizeof(ge_precomp) * 32 * 8));
    free(host_base);
    printf("Precomputed tables loaded to Constant Memory.\n");
}

// --- Configuration ---

#define BLOCK_SIZE 256
// Reduced from 4096 to 64 because SHA512+BaseMult is much heavier than PointAdd
#define ATTEMPTS_PER_BATCH 64
#define BATCH_SIZE ATTEMPTS_PER_BATCH
#define RING_BUFFER_SIZE 1024
#define RING_BUFFER_MASK (RING_BUFFER_SIZE - 1)

// --- Data Structures ---

struct Candidate {
    uint64_t thread_id;
    uint64_t iter_count;
};

struct DeviceRingBuffer {
    uint32_t write_head;
    Candidate items[RING_BUFFER_SIZE];
};

struct Stats {
    unsigned long long total_hashes;
};

// --- Globals & Device Functions ---

// Store target prefix indices for GPU filtering
__constant__ int c_prefix_indices[16];
__constant__ int c_prefix_len;

__device__ bool check_prefix(const unsigned char* s) {
    if (c_prefix_len == 0) return true;

    // 1. Count leading zeros
    int zcount = 0;
    while (zcount < 32 && !s[zcount]) zcount++;

    // Edge case: All zeros -> "1"
    if (zcount == 32) {
        if (c_prefix_len == 1 && c_prefix_indices[0] == 0) return true;
        return false;
    }

    // 2. Base conversion (Base58)
    // Max Base58 len for 32 bytes is ~44 chars, plus potential leading '1's.
    // Buffer size 50 is sufficient.
    unsigned char buf[50];

    // Initialize buf to 0
    #pragma unroll
    for(int k=0; k<50; k++) buf[k] = 0;

    int size = 50;
    int high = size - 1;

    // Perform "Multiply by 256 and add" algorithm
    for (int i = zcount; i < 32; ++i) {
        int carry = s[i];
        int j = size - 1;
        // Optimization: only iterate from current high mark
        while (j > high || carry) {
            int val = (int)buf[j] * 256 + carry;
            buf[j] = (unsigned char)(val % 58);
            carry = val / 58;
            j--;
            if (j < 0) break;
        }
        high = j;
    }

    // 3. Compare with prefix
    // Find start of non-zero digits in buf
    int buf_start = 0;
    while (buf_start < size && buf[buf_start] == 0) buf_start++;

    for (int k = 0; k < c_prefix_len; k++) {
        int val;
        if (k < zcount) {
            val = 0; // '1' is represented by index 0 in Base58
        } else {
            int offset = k - zcount;
            int buf_idx = buf_start + offset;

            if (buf_idx >= size) return false; // Prefix longer than result
            val = buf[buf_idx];
        }

        if (val != c_prefix_indices[k]) return false;
    }

    return true;
}


// --- Kernel ---

__global__ __launch_bounds__(BLOCK_SIZE, 1)
void phase1_filter_kernel(
    ge_p3 base_P,           // Unused
    uint64_t random_offset,
    DeviceRingBuffer* ring,
    Stats* stats
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t total_threads = gridDim.x * blockDim.x;

    // Correct Stride Logic:
    // We want each thread to check a unique sequence of seeds.
    // Seed = random_offset + tid + (global_iteration * total_threads)
    // start_index includes tid.
    uint64_t start_index = random_offset + (uint64_t)tid;

    uint64_t local_iter = 0;

    while (true) {
        for (int attempt = 0; attempt < ATTEMPTS_PER_BATCH; attempt++) {
            // Calculate current iteration count (how many steps this thread has taken)
            uint64_t current_thread_step = local_iter + attempt;

            // Calculate current seed value
            // current_val = start_index + (current_thread_step * total_threads)
            // This strides the search space by total_threads
            uint64_t current_val = start_index + (current_thread_step * (uint64_t)total_threads);

            unsigned char seed[32];
            #pragma unroll
            for(int i=0; i<32; i++) seed[i] = 0;

            uint64_t t_temp = current_val;

            // Populate seed with the 64-bit counter (little-endian)
            for(int i=0; i<8; i++) {
                seed[i] = t_temp & 0xFF;
                t_temp >>= 8;
            }

            // 2. SHA512(seed)
            unsigned char hash[64];
            sha512(seed, 32, hash);

            // 3. Clamp (Ed25519)
            hash[0] &= 248;
            hash[31] &= 63;
            hash[31] |= 64;

            // 4. Scalar Multiply (P = hash * Base)
            ge_p3 P;
            ge_scalarmult_base(&P, hash); // Uses first 32 bytes of hash as scalar

            // 5. Convert to Bytes (Public Key)
            fe recip;
            fe_invert(recip, P.Z);

            fe y;
            fe_mul(y, P.Y, recip);

            fe x;
            fe_mul(x, P.X, recip);
            int sign = fe_isnegative(x);

            unsigned char s[32];
            fe_tobytes(s, y);
            s[31] ^= (sign << 7);

            // 6. Check Prefix
            if (check_prefix(s)) {
                uint32_t slot = atomicAdd(&ring->write_head, 1);
                Candidate c;
                c.thread_id = tid;
                c.iter_count = current_thread_step; // Send thread-local step count
                ring->items[slot & RING_BUFFER_MASK] = c;
            }
        }

        local_iter += ATTEMPTS_PER_BATCH;

        if (threadIdx.x == 0) {
            atomicAdd((unsigned long long*)&stats->total_hashes, (unsigned long long)(ATTEMPTS_PER_BATCH * blockDim.x));
        }
    }
}

// --- Host Helper Functions ---

const char* B58_DIGITS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

int b58_index_host(char c) {
    const char *p = strchr(B58_DIGITS, c);
    if (p) return p - B58_DIGITS;
    return -1;
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
    uint8_t buf[200];
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

// --- Host Logic: Phase 2 ---

void phase2_solve(uint64_t thread_id, uint64_t iter_count, uint32_t total_threads, uint64_t random_offset, const char* suffix_check, const char* prefix_check) {
    unsigned __int128 offset = (unsigned __int128)random_offset +
                               ((unsigned __int128)thread_id);

    // Reconstruction logic matches Kernel:
    // current_val = offset + (iter_count * total_threads)

    unsigned __int128 current_val = offset + ((unsigned __int128)iter_count * (unsigned __int128)total_threads);

    unsigned char seed[32];
    memset(seed, 0, 32);

    unsigned __int128 temp = current_val;
    for(int i=0; i<8; i++) {
        seed[i] = (unsigned char)(temp & 0xFF);
        temp >>= 8;
    }

    // SHA512 + Clamp + BaseMult (Same as kernel)
    unsigned char hash[64];
    sha512(seed, 32, hash);

    hash[0] &= 248;
    hash[31] &= 63;
    hash[31] |= 64;

    ge_p3 A;
    ge_scalarmult_base(&A, hash);

    unsigned char publick[32];
    ge_p3_tobytes(publick, &A);

    char b58[128];
    size_t b58len = 128;
    b58enc(b58, &b58len, publick, 32);

    // EXACT VERIFICATION (The Authority)
    if (prefix_check && strlen(prefix_check) > 0) {
        if (strncmp(b58, prefix_check, strlen(prefix_check)) == 0) {
             printf("KEY FOUND\n");
             printf("{\"found\": true, \"public_key\": \"%s\", \"secret_key\": [", b58);
             for(int i=0; i<32; i++) printf("%d, ", seed[i]);
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

    printf("{\"found\": true, \"public_key\": \"%s\", \"secret_key\": [", b58);
    for(int i=0; i<32; i++) printf("%d, ", seed[i]);
    for(int i=0; i<31; i++) printf("%d, ", publick[i]);
    printf("%d]}\n", publick[31]);
    fflush(stdout);
    exit(0);
}

int main(int argc, char** argv) {
    const char* suffix = NULL;
    const char* prefix_str = NULL;
    int device = 0;

    for(int i=1; i<argc; i++) {
        if (strcmp(argv[i], "--suffix")==0 && i+1<argc) suffix = argv[i+1];
        if (strcmp(argv[i], "--prefix-str")==0 && i+1<argc) prefix_str = argv[i+1];
        if (strcmp(argv[i], "--prefix")==0 && i+1<argc) prefix_str = argv[i+1];
        if (strcmp(argv[i], "--device")==0 && i+1<argc) device = atoi(argv[i+1]);
        if (strcmp(argv[i], "--gpu-index")==0 && i+1<argc) device = atoi(argv[i+1]);
        if (strcmp(argv[i], "--generate-tables")==0) {
            generate_tables();
            return 0;
        }
        // Removed min/max limit args
    }

    int host_prefix_len = 0;
    int host_prefix_indices[16]; // Max prefix size support 16

    if (prefix_str) {
        host_prefix_len = strlen(prefix_str);
        if (host_prefix_len > 16) {
            fprintf(stderr, "Error: Prefix too long (max 16)\n");
            exit(1);
        }
        for (int i=0; i<host_prefix_len; i++) {
            int idx = b58_index_host(prefix_str[i]);
            if (idx < 0) {
                fprintf(stderr, "Error: Invalid Base58 character '%c'\n", prefix_str[i]);
                exit(1);
            }
            host_prefix_indices[i] = idx;
        }
        printf("Target Prefix: %s (Len: %d)\n", prefix_str, host_prefix_len);
    }

    cudaSetDevice(device);
    load_tables();

    // Copy Prefix Data to Constant Memory
    CHECK_CUDA(cudaMemcpyToSymbol(c_prefix_len, &host_prefix_len, sizeof(int)));
    if (host_prefix_len > 0) {
        CHECK_CUDA(cudaMemcpyToSymbol(c_prefix_indices, host_prefix_indices, sizeof(int) * host_prefix_len));
    }

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
    int gridSize = prop.multiProcessorCount * 128;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    uint64_t random_offset = dis(gen);

    printf("VanityForge Iterator (Probabilistic Filter)\n");
    printf("GPU: %s\n", prop.name);
    printf("Random Offset: %lu\n", random_offset);

    ge_p3 dummy;

    phase1_filter_kernel<<<gridSize, BLOCK_SIZE>>>(
        dummy,
        random_offset,
        d_ring,
        d_stats
    );

    CHECK_CUDA(cudaGetLastError());

    uint32_t local_read_head = 0;
    unsigned long long last_hashes = 0;
    auto last_time = std::chrono::high_resolution_clock::now();
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
