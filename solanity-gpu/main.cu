/*
 * VanityForge - Phase 1: High-Throughput Seed Grinder
 * Target: NVIDIA L4 (SM89)
 *
 * Architecture:
 * - Phase 1 (GPU): Seed Iterator -> SHA512 -> Clamp -> Point Mult
 * - "Correct" derivation for filtering (Matches Ed25519 standard).
 * - Ring Buffer: Transfers 'seed index' to Host.
 * - Phase 2 (Host): Reconstructs Seed.
 * - Verifies against "Correct" Ed25519 derivation (Double check).
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
#include "fe.h" // Include header first
#include "fe_ptx.cuh" // Include PTX implementation

// --- PTX Math Replacement Macro Logic ---
// We want to force ge.cu to use fe_mul_ptx/fe_sq_ptx when compiling for device.
// ge.cu uses 'fe_mul' and 'fe_sq'.
// We define macros to redirect these calls.
// Note: fe.cu implementation of fe_mul/fe_sq has __device__ __host__ qualifiers.
// If we define macro, we mask the function declaration in fe.cu if we include it after.
// But we need fe.cu for host-side execution and for other functions (fe_add, etc).
// The only way to cleanly swap is if we use the macro only for ge.cu?
// Or we can rely on C++ overloading? No, signatures are identical.
//
// STRATEGY:
// 1. Include fe.cu (provides standard fe_mul implementation).
// 2. Define macros `fe_mul` -> `fe_mul_ptx` ONLY for __CUDA_ARCH__.
//    But preprocessor runs before compilation.
//    If we do `#define fe_mul fe_mul_ptx`, then host code also tries to use fe_mul_ptx.
//    fe_mul_ptx is `__device__` only. Host compilation will fail.
//
//    Correct approach:
//    Rename standard functions in fe.cu? No, I can't touch fe.cu easily.
//
//    Wait, I can include `fe.cu` first.
//    Then define macros.
//    Then include `ge.cu`.
//
//    `fe.cu`: `void fe_mul(...) {...}` defined.
//    `ge.cu` calls `fe_mul(...)`.
//    If I define `#define fe_mul(h,f,g) fe_mul_ptx(h,f,g)` before including `ge.cu`,
//    then `ge.cu` will call `fe_mul_ptx` inside its functions.
//
//    However, this macro applies to both Host and Device code in `ge.cu`.
//    `ge.cu` functions are `__host__ __device__`.
//    If they call `fe_mul_ptx`, it must be available on Host. It is NOT.
//
//    So we need a dispatcher?
//
//    `__device__ __forceinline__ void fe_mul_dispatch(fe h, const fe f, const fe g) { fe_mul_ptx(h,f,g); }`
//    `__host__ void fe_mul_dispatch(fe h, const fe f, const fe g) { fe_mul(h,f,g); }` (calls original)
//
//    But `fe_mul` (original) is defined in `fe.cu`.
//
//    If I define:
//    ```
//    #ifdef __CUDA_ARCH__
//      #define fe_mul fe_mul_ptx
//      #define fe_sq fe_sq_ptx
//    #endif
//    ```
//    Macros don't respect `__CUDA_ARCH__` (which is a compiler define, yes, but preprocessor sees it during device compilation pass).
//    NVCC splits compilation.
//    During device pass, `__CUDA_ARCH__` is defined.
//    During host pass, it is not.
//
//    So yes, wrapping macros in `#ifdef __CUDA_ARCH__` works!
//
//    BUT, `fe.cu` is included. `fe_mul` is defined as a symbol.
//    If I `#define fe_mul ...`, I am shadowing the function name with a macro.
//    Calls in `ge.cu` will use the macro.
//    This is what we want for device pass.
//
//    So:
//    1. Include `fe.cu` (defines standard functions).
//    2. Include `fe_ptx.cuh` (defines ptx functions).
//    3. `#ifdef __CUDA_ARCH__`
//       `#define fe_mul fe_mul_ptx`
//       `#define fe_sq fe_sq_ptx`
//       `#endif`
//    4. Include `ge.cu` (calls `fe_mul` -> macro -> `fe_mul_ptx` on device).
//
//    This works perfectly.

#include "fe.cu"

#ifdef __CUDA_ARCH__
    #define fe_mul fe_mul_ptx
    #define fe_sq fe_sq_ptx
#endif

#include "ge.h"           
#include "precomp_data.h" 
#include "ge.cu"

// Undefine to avoid messing up subsequent host code (though main is last)
#ifdef __CUDA_ARCH__
    #undef fe_mul
    #undef fe_sq
#endif

// Host-side SHA512 (for verification)
#include "sha512.cu" 

// GPU-side Optimized SHA512 (for grinding)
#include "sha512_ed25519.cuh"

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

    CHECK_CUDA(cudaMemcpyToSymbol(d_base, host_base, sizeof(ge_precomp) * 32 * 8));
    free(host_base);
    printf("Precomputed tables loaded to Global Device Memory.\n");
}

// --- Configuration ---

#define BLOCK_SIZE 256
#define ATTEMPTS_PER_BATCH 256
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
    volatile uint32_t read_head;
    Candidate items[RING_BUFFER_SIZE];
};

struct Stats {
    unsigned long long total_hashes;
};

// --- Globals & Device Functions ---

// Store target prefix indices for GPU filtering
__constant__ int c_prefix_indices[16];
__constant__ int c_prefix_len;
// Store target suffix indices for GPU filtering
__constant__ int c_suffix_indices[16];
__constant__ int c_suffix_len;

__device__ bool check_match(const unsigned char* s) {
    // 1. Count leading zeros
    int zcount = 0;
    while (zcount < 32 && !s[zcount]) zcount++;

    // Edge case: All zeros -> "1"
    if (zcount == 32) {
        if (c_prefix_len == 1 && c_prefix_indices[0] == 0) return true;
        return false;
    }

    // 2. Base conversion (Base58)
    unsigned char buf[50];
    #pragma unroll
    for(int k=0; k<50; k++) buf[k] = 0;

    int size = 50;
    int high = size - 1;

    for (int i = zcount; i < 32; ++i) {
        int carry = s[i];
        int j = size - 1;
        while (j > high || carry) {
            int val = (int)buf[j] * 256 + carry;
            buf[j] = (unsigned char)(val % 58);
            carry = val / 58;
            j--;
            if (j < 0) break;
        }
        high = j;
    }

    int buf_start = 0;
    while (buf_start < size && buf[buf_start] == 0) buf_start++;

    // Prefix Check
    if (c_prefix_len > 0) {
        for (int k = 0; k < c_prefix_len; k++) {
            int val;
            if (k < zcount) {
                val = 0;
            } else {
                int offset = k - zcount;
                int buf_idx = buf_start + offset;
                if (buf_idx >= size) return false;
                val = buf[buf_idx];
            }
            if (val != c_prefix_indices[k]) return false;
        }
    }

    // Suffix Check
    if (c_suffix_len > 0) {
        for (int k = 0; k < c_suffix_len; k++) {
            // Compare end of buffer against suffix indices
            if (buf[50 - c_suffix_len + k] != c_suffix_indices[k]) return false;
        }
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

    uint64_t start_index = random_offset + (uint64_t)tid;
    uint64_t local_iter = 0;

    unsigned char seed[32];
    unsigned char hash[64];

    while (true) {
        for (int attempt = 0; attempt < ATTEMPTS_PER_BATCH; attempt++) {
            uint64_t current_thread_step = local_iter + attempt;
            uint64_t current_val = start_index + (current_thread_step * (uint64_t)total_threads);

            // 1. Generate Seed (Random Counter)
            #pragma unroll
            for(int i=0; i<32; i++) seed[i] = 0;
            
            uint64_t t_temp = current_val;
            for(int i=0; i<8; i++) {
                seed[i] = t_temp & 0xFF;
                t_temp >>= 8;
            }

            // 2. Hash Seed (SHA-512) - GPU Optimized
            // 
            sha512_32byte_seed(seed, hash);

            // 3. Clamp Hash (Ed25519 Standard)
            hash[0] &= 248;
            hash[31] &= 63;
            hash[31] |= 64;

            // 4. Point Multiplication (Scalar * Base Point)
            ge_p3 P;

            // This calls ge_scalarmult_base.
            // Because we used macros before including ge.cu, on Device logic this will use fe_mul_ptx!
            ge_scalarmult_base(&P, hash);

            // 5. Convert to Bytes (Public Key)
            fe recip;
            fe_invert(recip, P.Z);
            fe y;
            fe_mul_ptx(y, P.Y, recip);
            fe x;
            fe_mul_ptx(x, P.X, recip);

            int sign = fe_isnegative(x);
            unsigned char s[32];
            fe_tobytes(s, y);
            s[31] ^= (sign << 7);

            // DEBUG: Print the first key from the first thread to verify tables are loaded
            if (tid == 0 && attempt == 0 && local_iter == 0) {
                printf("[DEBUG] GPU GENERATED KEY: ");
                for(int k=0; k<32; k++) printf("%02x", s[k]);
                printf("\n");
            }

            // 6. Check Match (Prefix + Suffix)
            if (check_match(s)) {
                // Backpressure: Drop if buffer > 50% full
                uint32_t rh = ring->read_head;
                uint32_t wh = ring->write_head;
                if ((wh - rh) < (RING_BUFFER_SIZE / 2)) {
                    uint32_t slot = atomicAdd(&ring->write_head, 1);
                    Candidate c;
                    c.thread_id = tid;
                    c.iter_count = current_thread_step;
                    ring->items[slot & RING_BUFFER_MASK] = c;
                }
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

bool phase2_solve(uint64_t thread_id, uint64_t iter_count, uint32_t total_threads, uint64_t random_offset, const char* suffix_check, const char* prefix_check) {
    unsigned __int128 offset = (unsigned __int128)random_offset +
                               ((unsigned __int128)thread_id);
    unsigned __int128 current_val = offset + ((unsigned __int128)iter_count * (unsigned __int128)total_threads);

    unsigned char seed[32];
    memset(seed, 0, 32);
    unsigned __int128 temp = current_val;
    for(int i=0; i<8; i++) {
        seed[i] = (unsigned char)(temp & 0xFF);
        temp >>= 8;
    }

    // 1. Calculate Derived Public Key (Standard Ed25519: Seed -> SHA512 -> Clamp -> P_derived)
    // Uses Host SHA512 implementation
    unsigned char hash[64];
    sha512(seed, 32, hash);
    hash[0] &= 248;
    hash[31] &= 63;
    hash[31] |= 64;

    ge_p3 P_derived;
    // Uses host-side standard math (no macros)
    ge_scalarmult_base(&P_derived, hash);
    unsigned char publick_derived[32];
    ge_p3_tobytes(publick_derived, &P_derived);

    // Verify prefix on derived (Source of Truth)
    char b58[128];
    size_t b58len = 128;
    b58enc(b58, &b58len, publick_derived, 32);

    if (prefix_check && strlen(prefix_check) > 0) {
        if (strncmp(b58, prefix_check, strlen(prefix_check)) != 0) {
             return false;
        }
    }

    if (suffix_check && strlen(suffix_check) > 0) {
        size_t len = strlen(b58);
        size_t slen = strlen(suffix_check);
        if (len < slen || strcmp(b58 + len - slen, suffix_check) != 0) {
             return false;
        }
    }

    // Encode seed as Base58 for JSON output
    char seed_b58[128];
    size_t seed_len = 128;
    b58enc(seed_b58, &seed_len, seed, 32);

    // Required JSON format
    printf("{\"found\": true, \"seed\": \"%s\", \"public_key\": \"%s\"}\n", seed_b58, b58);
    fflush(stdout);
    return true; // Found and valid
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
    }

    int host_prefix_len = 0;
    int host_prefix_indices[16];
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

    int host_suffix_len = 0;
    int host_suffix_indices[16];
    if (suffix) {
        host_suffix_len = strlen(suffix);
        if (host_suffix_len > 4) {
            fprintf(stderr, "Error: Suffix too long (max 4)\n");
            exit(1);
        }
        for (int i=0; i<host_suffix_len; i++) {
            int idx = b58_index_host(suffix[i]);
            if (idx < 0) {
                fprintf(stderr, "Error: Invalid Base58 character '%c'\n", suffix[i]);
                exit(1);
            }
            host_suffix_indices[i] = idx;
        }
        printf("Target Suffix: %s (Len: %d)\n", suffix, host_suffix_len);
    }

    cudaSetDevice(device);
    load_tables();

    // Limit GPU to max 6 chars for prefix (was 5), and 4 for suffix
    int gpu_prefix_len = host_prefix_len;
    if (gpu_prefix_len > 6) gpu_prefix_len = 6;

    int gpu_suffix_len = host_suffix_len;
    if (gpu_suffix_len > 4) gpu_suffix_len = 4;

    CHECK_CUDA(cudaMemcpyToSymbol(c_prefix_len, &gpu_prefix_len, sizeof(int)));
    // We still copy all indices, but kernel only uses gpu_prefix_len
    if (host_prefix_len > 0) {
        CHECK_CUDA(cudaMemcpyToSymbol(c_prefix_indices, host_prefix_indices, sizeof(int) * host_prefix_len));
    }

    CHECK_CUDA(cudaMemcpyToSymbol(c_suffix_len, &gpu_suffix_len, sizeof(int)));
    if (host_suffix_len > 0) {
        CHECK_CUDA(cudaMemcpyToSymbol(c_suffix_indices, host_suffix_indices, sizeof(int) * host_suffix_len));
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

    printf("VanityForge Iterator (Full Ed25519 Mode + PTX Math)\n");
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
            bool found = phase2_solve(c.thread_id, c.iter_count, total_threads, random_offset, suffix, prefix_str);
            if (found) {
                return 0;
                // Exit cleanly on success
            }
            local_read_head++;
            if (write_head - local_read_head > RING_BUFFER_SIZE) {
                local_read_head = write_head;
                printf("[Warn] Ring Buffer Overflow!\n");
            }
        }

        // Release backpressure
        h_ring_mapped->read_head = local_read_head;
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
