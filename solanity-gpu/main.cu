#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <ctime>
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "curand_kernel.h"
#include "ed25519.h"
#include "fixedint.h"
#include "gpu_common.h"
#include "gpu_ctx.h"
#include "gpu_ctx.cu"
#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"
#include "verify.cu"

#define MAX_ITERATIONS 2000000000
#define STOP_AFTER_KEYS_FOUND 1
#define ATTEMPTS_PER_EXECUTION 512

// Structure to report results from Device to Host
struct SearchResult {
    int found;
    unsigned char pubkey[32];
    unsigned char seed[32];
};

// Structure to pass target range to Device
struct Range {
    uint64_t min64[4];
    uint64_t max64[4];
};

__device__ __constant__ Range d_range;

typedef struct {
    curandState* states;
    int gridSize;
    int blockSize;
    SearchResult* result; // Device pointer
} config;

// --- Host-Side Base58 Helpers ---

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

int b58dec_host(uint8_t *bin, size_t binsz, const char *b58) {
    size_t i, j;
    size_t len = strlen(b58);
    const char *b58digits = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    int b58map[256];
    memset(b58map, -1, sizeof(b58map));
    for (i = 0; i < 58; i++) b58map[(uint8_t)b58digits[i]] = i;

    uint8_t buf[64]; // Temporary buffer for big integer arithmetic
    memset(buf, 0, sizeof(buf));

    for (i = 0; i < len; i++) {
        int c = (unsigned char)b58[i];
        if (b58map[c] == -1) return -1; // Invalid char
        uint32_t carry = b58map[c];
        for (j = 63; ; j--) {
            uint32_t val = buf[j] * 58 + carry;
            buf[j] = val & 0xFF;
            carry = val >> 8;
            if (j == 0) break;
        }
    }

    // Check for overflow beyond binsz
    // We are decoding to fixed 32 bytes (binsz).
    // The number in 'buf' (64 bytes) must fit in last 'binsz' bytes.
    for(i = 0; i < 64 - binsz; i++) {
        if (buf[i] != 0) return -2; // Overflow
    }

    // Copy to output
    for(i = 0; i < binsz; i++) {
        bin[binsz - 1 - i] = buf[63 - i];
    }
    return 0;
}

// Logic Fix 1: Smart Fallback
// Try 44 chars, if overflow, try 43 chars.
void compute_prefix_range(const char* prefix, uint8_t target_min[32], uint8_t target_max[32]) {
    char min_s[64];
    char max_s[64];
    size_t len = strlen(prefix);

    // Default to 44, but prepare to fallback to 43 if 44 overflows
    size_t target_lens[] = {44, 43};
    bool success = false;

    for (int k = 0; k < 2; k++) {
        size_t target_len = target_lens[k];
        if (len > target_len) continue;

        // Construct Min String: Prefix + '1's
        strcpy(min_s, prefix);
        for(size_t i=len; i<target_len; i++) min_s[i] = '1';
        min_s[target_len] = 0;

        // Construct Max String: Prefix + 'z's
        strcpy(max_s, prefix);
        for(size_t i=len; i<target_len; i++) max_s[i] = 'z';
        max_s[target_len] = 0;

        // Try Decode Min. Trust b58dec_host return codes (0=success).
        int err1 = b58dec_host(target_min, 32, min_s);
        if (err1 == 0) {
            // Min is valid! Now decode Max.
            int err2 = b58dec_host(target_max, 32, max_s);
            if (err2 != 0) {
                // If max overflows, cap at 0xFF...FF
                memset(target_max, 0xFF, 32);
            }
            success = true;
            break; // Found a valid range length
        }
        // If err1 != 0 (e.g. -2 Overflow), we loop to try 43
    }

    if (!success) {
        // Both 44 and 43 failed (or prefix was too long)
        // Set to impossible range to trigger error in main
        memset(target_min, 0xFF, 32);
        memset(target_max, 0x00, 32);
    }
}

// Prototypes
void vanity_setup(config& vanity, int gpu_index);
void vanity_run(config& vanity, int gpu_index, Range range, const char* prefix_str, const char* suffix_str);
__global__ void vanity_init(unsigned long long int* seed, curandState* state);
// Fix B: Relax launch bounds to 256, 1
__global__ __launch_bounds__(256, 1) void vanity_scan(curandState* state, SearchResult* result, int* execution_count);

int main(int argc, char const* argv[]) {
    // 1. Device Capability Check
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU name: %s | Compute Capability: %d.%d\n",
           prop.name, prop.major, prop.minor);
    fflush(stdout);

    ed25519_set_verbose(false);

    char* prefix_str = NULL;
    char* suffix_str = NULL;
    int gpu_index = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prefix") == 0 && i + 1 < argc) {
            prefix_str = (char*)argv[++i];
        } else if (strcmp(argv[i], "--suffix") == 0 && i + 1 < argc) {
            suffix_str = (char*)argv[++i];
        } else if (strcmp(argv[i], "--gpu-index") == 0 && i + 1 < argc) {
            gpu_index = atoi(argv[++i]);
        }
    }

    uint64_t min_buf[4] = {0};
    uint64_t max_buf[4];
    memset(max_buf, 0xFF, 32); // Default max

    if (prefix_str) {
        compute_prefix_range(prefix_str, (uint8_t*)min_buf, (uint8_t*)max_buf);
    }

    // Check for reversed range (impossible prefix) using 8-bit arrays before swap
    bool range_valid = false;
    uint8_t* p_min = (uint8_t*)min_buf;
    uint8_t* p_max = (uint8_t*)max_buf;

    for(int i=0; i<32; i++) {
        if (p_min[i] < p_max[i]) { range_valid = true; break; }
        if (p_min[i] > p_max[i]) { range_valid = false; break; }
    }
    // If equal, it's valid (1 key)

    if (!range_valid && memcmp(min_buf, max_buf, 32) != 0) {
         printf("Error: Impossible prefix range.\n");
         exit(1);
    }

    // Convert to Range struct with pre-swapped endianness
    Range range;
    for (int i = 0; i < 4; i++) {
        range.min64[i] = __builtin_bswap64(min_buf[i]);
        range.max64[i] = __builtin_bswap64(max_buf[i]);
    }

    config vanity;
    vanity_setup(vanity, gpu_index);
    vanity_run(vanity, gpu_index, range, prefix_str, suffix_str);
    return 0;
}

unsigned long long int makeSeed() {
    unsigned long long int seed = 0;
    char *pseed = (char *)&seed;
    std::random_device rd;
    for(unsigned int b=0; b<sizeof(seed); b++) {
      auto r = rd();
      char *entropy = (char *)&r;
      pseed[b] = entropy[0];
    }
    return seed;
}

void vanity_setup(config &vanity, int gpu_index) {
    CUDA_CHK(cudaSetDevice(gpu_index));

    cudaDeviceProp device;
    CUDA_CHK(cudaGetDeviceProperties(&device, gpu_index));

    int minGridSize, bestBlockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, vanity_scan, 0, 0);

    int numBlocks = device.multiProcessorCount * 32;
    if (numBlocks > 65535) numBlocks = 65535;

    vanity.gridSize = numBlocks;
    vanity.blockSize = bestBlockSize;
    if (vanity.blockSize < 32) vanity.blockSize = 32;
    if (vanity.blockSize > 1024) vanity.blockSize = 1024;

    size_t totalThreads = (size_t)vanity.gridSize * (size_t)vanity.blockSize;

    // Check memory for curand states
    size_t req_bytes = totalThreads * sizeof(curandState);
    size_t freeMem = 0, totalMem = 0;
#if CUDART_VERSION >= 10000
    cudaMemGetInfo(&freeMem, &totalMem);
#endif
    if (freeMem > 0 && req_bytes > (freeMem * 8) / 10) {
        size_t allowedThreads = (freeMem * 8) / 10 / sizeof(curandState);
        int new_blocks = (int)(allowedThreads / vanity.blockSize);
        if (new_blocks < 1) new_blocks = 1;
        vanity.gridSize = new_blocks;
        totalThreads = (size_t)vanity.gridSize * (size_t)vanity.blockSize;
        req_bytes = totalThreads * sizeof(curandState);
    }

    printf("GPU: %s | SMs=%d | blockSize=%d | gridSize=%d | totalThreads=%zu\n",
           device.name, device.multiProcessorCount, vanity.blockSize, vanity.gridSize, totalThreads);
    fflush(stdout);

    unsigned long long int rseed = makeSeed();
    unsigned long long int* dev_rseed;
    CUDA_CHK(cudaMalloc((void**)&dev_rseed, sizeof(unsigned long long int)));
    CUDA_CHK(cudaMemcpy(dev_rseed, &rseed, sizeof(unsigned long long int), cudaMemcpyHostToDevice));

    CUDA_CHK(cudaMalloc((void**)&vanity.states, totalThreads * sizeof(curandState)));
    CUDA_CHK(cudaMalloc((void**)&vanity.result, sizeof(SearchResult)));
    CUDA_CHK(cudaMemset(vanity.result, 0, sizeof(SearchResult)));

    vanity_init<<<vanity.gridSize, vanity.blockSize>>>(dev_rseed, vanity.states);
    CUDA_CHK(cudaDeviceSynchronize());

    cudaFree(dev_rseed); // Cleanup seed

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error (init): %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

void vanity_run(config &vanity, int gpu_index, Range range, const char* prefix_str, const char* suffix_str) {
    CUDA_CHK(cudaSetDevice(gpu_index));
    CUDA_CHK(cudaMemcpyToSymbol(d_range, &range, sizeof(Range)));

    int* dev_executions_this_gpu;
    CUDA_CHK(cudaMalloc((void**)&dev_executions_this_gpu, sizeof(int)));

    unsigned long long total_checked = 0;
    size_t totalThreads = (size_t)vanity.gridSize * (size_t)vanity.blockSize;
    SearchResult host_result;

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        // Launch Kernel
        vanity_scan<<<vanity.gridSize, vanity.blockSize>>>(vanity.states, vanity.result, dev_executions_this_gpu);

        CUDA_CHK(cudaDeviceSynchronize());

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
            exit(1);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        double seconds = diff.count();
        unsigned long long current_batch = (unsigned long long)totalThreads * ATTEMPTS_PER_EXECUTION;
        total_checked += current_batch;

        double speed_mh = 0.0;
        if (seconds > 0) speed_mh = (double)current_batch / seconds / 1000000.0;

        printf("Speed: %.2f MH/s | Total checked: %.2f B\n", speed_mh, (double)total_checked / 1000000000.0);
        fflush(stdout);

        // Check result
        CUDA_CHK(cudaMemcpy(&host_result, vanity.result, sizeof(SearchResult), cudaMemcpyDeviceToHost));
        if (host_result.found) {
            // Match found by GPU (Prefix confirmed)
            // Perform full verification on Host
            char b58_key[256];
            size_t b58_len = 256;
            b58enc_host(b58_key, &b58_len, host_result.pubkey, 32);

            // Verify Suffix (if any)
            bool match = true;
            if (suffix_str && strlen(suffix_str) > 0) {
                size_t key_len = strlen(b58_key);
                size_t suf_len = strlen(suffix_str);
                if (key_len < suf_len) match = false;
                else {
                    if (strcmp(b58_key + key_len - suf_len, suffix_str) != 0) match = false;
                }
            }

            if (match) {
                // Output JSON
                printf("{\"public_key\": \"%s\", \"secret_key\": [", b58_key);
                for(int n=0; n<32; n++) {
                    printf("%d,", host_result.seed[n]);
                }
                for(int n=0; n<32; n++) {
                    printf("%d%s", host_result.pubkey[n], (n==31 ? "" : ","));
                }
                printf("]}\n");
                fflush(stdout);
                exit(0);
            } else {
                // False positive (suffix didn't match), reset and continue
                // Note: We might have missed other matches in this batch, but with high speed/randomness it's acceptable.
                CUDA_CHK(cudaMemset(vanity.result, 0, sizeof(SearchResult)));
            }
        }
    }
}

// 1. Add Device-safe byte swap helper
__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t lo = (uint32_t)(x & 0xFFFFFFFF);

    hi = __byte_perm(hi, 0, 0x0123);
    lo = __byte_perm(lo, 0, 0x0123);

    return ((uint64_t)lo << 32) | hi;
}

__global__ void vanity_init(unsigned long long int* rseed, curandState* state) {
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    curand_init(*rseed + id, id, 0, &state[id]);
}

// Fix B: Relax launch bounds to 256, 1
__global__ __launch_bounds__(256, 1) void vanity_scan(curandState* state, SearchResult* result, int* exec_count) {
    int id = threadIdx.x + (blockIdx.x * blockDim.x);

    ge_p3 A;
    curandState localState = state[id];

    __align__(8) unsigned char seed[32] = {0};
    __align__(8) unsigned char publick[32] = {0};
    unsigned char privatek[64] = {0};

    // Initialize seed from state
    for (int i = 0; i < 32; ++i) {
        float random = curand_uniform(&localState);
        seed[i] = (uint8_t)(random * 255);
    }

    sha512_context md;

    // Fix A: Remove #pragma unroll for the main loop
    for (int attempts = 0; attempts < ATTEMPTS_PER_EXECUTION; ++attempts) {
        // --- SHA512 ---
        md.curlen = 0; md.length = 0;
        md.state[0] = UINT64_C(0x6a09e667f3bcc908);
        md.state[1] = UINT64_C(0xbb67ae8584caa73b);
        md.state[2] = UINT64_C(0x3c6ef372fe94f82b);
        md.state[3] = UINT64_C(0xa54ff53a5f1d36f1);
        md.state[4] = UINT64_C(0x510e527fade682d1);
        md.state[5] = UINT64_C(0x9b05688c2b3e6c1f);
        md.state[6] = UINT64_C(0x1f83d9abfb41bd6b);
        md.state[7] = UINT64_C(0x5be0cd19137e2179);

        // seed is 32 bytes
        const unsigned char *in = seed;
        #pragma unroll
        for (size_t i = 0; i < 32; i++) md.buf[i] = in[i];
        md.curlen = 32;

        md.length = 256; // 32 * 8
        md.buf[32] = 0x80;
        // Pad with zeros up to 120
        #pragma unroll
        for (int i = 33; i < 120; i++) md.buf[i] = 0;
        STORE64H(md.length, md.buf+120);

        uint64_t S[8], W[80], t0, t1;
        int i;
        #pragma unroll
        for (i = 0; i < 8; i++) S[i] = md.state[i];
        #pragma unroll
        for (i = 0; i < 16; i++) LOAD64H(W[i], md.buf + (8*i));
        #pragma unroll
        for (i = 16; i < 80; i++) W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];

        #define RND(a,b,c,d,e,f,g,h,i) \
        t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i]; \
        t1 = Sigma0(a) + Maj(a, b, c);\
        d += t0; \
        h  = t0 + t1;

        #pragma unroll
        for (i = 0; i < 80; i += 8) {
            RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],i+0);
            RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],i+1);
            RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],i+2);
            RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],i+3);
            RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],i+4);
            RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],i+5);
            RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],i+6);
            RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],i+7);
        }
        #undef RND

        #pragma unroll
        for (i = 0; i < 8; i++) md.state[i] = md.state[i] + S[i];
        #pragma unroll
        for (i = 0; i < 8; i++) STORE64H(md.state[i], privatek+(8*i));

        privatek[0]  &= 248;
        privatek[31] &= 63;
        privatek[31] |= 64;

        ge_scalarmult_base(&A, privatek);
        ge_p3_tobytes(publick, &A);

        const uint64_t* pub64 = reinterpret_cast<const uint64_t*>(publick);
        // range.min64 and range.max64 are already aligned uint64_t[4] in the struct

        // 1. Fast Fail
        // Swap bytes to ensure Big-endian lexicographical comparison on Little-endian GPU
        uint64_t p0 = bswap64(pub64[0]);
        // d_range values are already swapped on host

        if (p0 < d_range.min64[0] || p0 > d_range.max64[0]) {
            goto increment_seed;
        }

        {
            // 2. Full Lexicographical Compare
            // Check >= Min
            int ge_min = 1;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint64_t p = bswap64(pub64[i]);
                if (p > d_range.min64[i]) break;
                if (p < d_range.min64[i]) { ge_min = 0; break; }
            }
            if (!ge_min) goto increment_seed;

            // Check <= Max
            int le_max = 1;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint64_t p = bswap64(pub64[i]);
                if (p < d_range.max64[i]) break;
                if (p > d_range.max64[i]) { le_max = 0; break; }
            }
            if (!le_max) goto increment_seed;
        }

        // Found a match!
        if (atomicCAS(&result->found, 0, 1) == 0) {
            #pragma unroll
            for(int k=0; k<32; k++) result->pubkey[k] = publick[k];
            #pragma unroll
            for(int k=0; k<32; k++) result->seed[k] = seed[k];
        }

increment_seed:
        // --- Vectorized Seed Increment ---
        uint64_t* seed64 = (uint64_t*)seed;
        #pragma unroll
        for(int k=0; k<4; k++) {
            seed64[k]++;
            if (seed64[k] != 0) break; // No overflow, done.
            // If overflow (wrapped to 0), continue to next word (carry).
        }
    }

    state[id] = localState;
}
