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
#define ATTEMPTS_PER_EXECUTION 16384

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
void compute_prefix_range(const char* prefix, uint8_t target_min[32], uint8_t target_max[32]) {
    char min_s[64];
    char max_s[64];
    size_t len = strlen(prefix);

    size_t target_lens[] = {44, 43};
    bool success = false;

    for (int k = 0; k < 2; k++) {
        size_t target_len = target_lens[k];
        if (len > target_len) continue;

        strcpy(min_s, prefix);
        for(size_t i=len; i<target_len; i++) min_s[i] = '1';
        min_s[target_len] = 0;

        // Attempt decode Min
        int err1 = b58dec_host(target_min, 32, min_s);
        if (err1 == 0) {
            // Success decoding min. Now try max.
            strcpy(max_s, prefix);
            for(size_t i=len; i<target_len; i++) max_s[i] = 'z';
            max_s[target_len] = 0;

            int err2 = b58dec_host(target_max, 32, max_s);
            if (err2 != 0) {
                // If max overflows, cap at max uint256
                memset(target_max, 0xFF, 32);
            }
            success = true;
            break;
        }
        // If err1 != 0 (e.g. overflow), loop continues to try next length (43)
    }

    if (!success) {
        memset(target_min, 0xFF, 32);
        memset(target_max, 0x00, 32);
    }
}

// Prototypes
void vanity_setup(config& vanity, int gpu_index);
void vanity_run(config& vanity, int gpu_index, Range range, const char* prefix_str, const char* suffix_str);
__global__ void vanity_init(unsigned long long int* seed, curandState* state);
__global__ __launch_bounds__(256, 1) void vanity_scan(curandState* state, SearchResult* result, int* execution_count);

int main(int argc, char const* argv[]) {
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
    memset(max_buf, 0xFF, 32);

    if (prefix_str) {
        compute_prefix_range(prefix_str, (uint8_t*)min_buf, (uint8_t*)max_buf);
    }

    bool range_valid = false;
    uint8_t* p_min = (uint8_t*)min_buf;
    uint8_t* p_max = (uint8_t*)max_buf;

    for(int i=0; i<32; i++) {
        if (p_min[i] < p_max[i]) { range_valid = true; break; }
        if (p_min[i] > p_max[i]) { range_valid = false; break; }
    }

    if (!range_valid && memcmp(min_buf, max_buf, 32) != 0) {
         printf("Error: Impossible prefix range.\n");
         exit(1);
    }

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

    cudaFree(dev_rseed);

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

        CUDA_CHK(cudaMemcpy(&host_result, vanity.result, sizeof(SearchResult), cudaMemcpyDeviceToHost));
        if (host_result.found) {
            char b58_key[256];
            size_t b58_len = 256;
            b58enc_host(b58_key, &b58_len, host_result.pubkey, 32);

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
                CUDA_CHK(cudaMemset(vanity.result, 0, sizeof(SearchResult)));
            }
        }
    }
}

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

__global__ __launch_bounds__(256, 1) void vanity_scan(curandState* state, SearchResult* result, int* execution_count) {
    int id = threadIdx.x + (blockIdx.x * blockDim.x);

    ge_p3 A;
    curandState localState = state[id];

    __align__(8) unsigned char seed[32] = {0};
    __align__(8) unsigned char publick[32] = {0};
    unsigned char privatek[64] = {0};

    // Optimization 3: Fast Seed Generation using curand() (uint32)
    uint32_t* seed32 = (uint32_t*)seed;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        seed32[i] = curand(&localState);
    }

    const uint64_t IV[8] = {
        UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
        UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
        UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
        UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)
    };

    for (int attempts = 0; attempts < ATTEMPTS_PER_EXECUTION; ++attempts) {
        // Periodic check to exit early if key found by another thread
        if ((attempts & 127) == 0) {
             if (result->found) break;
        }

        // Optimization 1: Rolling SHA-512 without sha512_context
        uint64_t S[8], W[16], t0, t1;

        // Init S from IV
        #pragma unroll
        for(int i=0; i<8; i++) S[i] = IV[i];

        // Init W from seed
        #pragma unroll
        for(int i=0; i<4; i++) {
             LOAD64H(W[i], seed + 8*i);
        }
        W[4] = UINT64_C(0x8000000000000000);
        #pragma unroll
        for(int i=5; i<15; i++) W[i] = 0;
        W[15] = 256; // Length = 256 bits

        #define RND(a,b,c,d,e,f,g,h,i,w) \
        t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + w; \
        t1 = Sigma0(a) + Maj(a, b, c);\
        d += t0; \
        h  = t0 + t1;

        // SHA Loop - 80 rounds
        // NO UNROLL on the outer loop to reduce code size
        for (int i = 0; i < 80; i += 8) {
            if (i >= 16) {
                #pragma unroll
                for(int j=0; j<8; j++) {
                     int k = i + j;
                     W[k&15] = Gamma1(W[(k+14)&15]) + W[(k+9)&15] + Gamma0(W[(k+1)&15]) + W[k&15];
                }
            }
            RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],i+0, W[(i+0)&15]);
            RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],i+1, W[(i+1)&15]);
            RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],i+2, W[(i+2)&15]);
            RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],i+3, W[(i+3)&15]);
            RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],i+4, W[(i+4)&15]);
            RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],i+5, W[(i+5)&15]);
            RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],i+6, W[(i+6)&15]);
            RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],i+7, W[(i+7)&15]);
        }
        #undef RND

        // Finalize: Add IV and Store to privatek
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint64_t h = S[i] + IV[i];
            STORE64H(h, privatek + 8*i);
        }

        privatek[0]  &= 248;
        privatek[31] &= 63;
        privatek[31] |= 64;

        ge_scalarmult_base(&A, privatek);
        ge_p3_tobytes(publick, &A);

        const uint64_t* pub64 = reinterpret_cast<const uint64_t*>(publick);

        // Optimization 2: Hoist bswap64
        uint64_t p_swapped[4];
        #pragma unroll
        for(int k=0; k<4; k++) p_swapped[k] = bswap64(pub64[k]);

        // Fast Fail
        uint64_t p0 = p_swapped[0];

        if (p0 < d_range.min64[0] || p0 > d_range.max64[0]) {
            goto increment_seed;
        }

        {
            // Full Lexicographical Compare using hoisted values
            int ge_min = 1;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint64_t p = p_swapped[i];
                if (p > d_range.min64[i]) break;
                if (p < d_range.min64[i]) { ge_min = 0; break; }
            }
            if (!ge_min) goto increment_seed;

            int le_max = 1;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint64_t p = p_swapped[i];
                if (p < d_range.max64[i]) break;
                if (p > d_range.max64[i]) { le_max = 0; break; }
            }
            if (!le_max) goto increment_seed;
        }

        if (atomicCAS(&result->found, 0, 1) == 0) {
            #pragma unroll
            for(int k=0; k<32; k++) result->pubkey[k] = publick[k];
            #pragma unroll
            for(int k=0; k<32; k++) result->seed[k] = seed[k];
        }

increment_seed:
        uint64_t* seed64 = (uint64_t*)seed;
        #pragma unroll
        for(int k=0; k<4; k++) {
            seed64[k]++;
            if (seed64[k] != 0) break;
        }
    }

    state[id] = localState;
}
