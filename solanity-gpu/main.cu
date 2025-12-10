#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <ctime>
#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "curand_kernel.h"
#include "ed25519.h"
#include "fixedint.h"
#include "gpu_common.h"
#include "gpu_ctx.h"

// Include implementation files directly as in original vanity.cu
#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"
#include "verify.cu"

// Constants replacing config.h
#define MAX_ITERATIONS 2000000000
#define STOP_AFTER_KEYS_FOUND 1
#define ATTEMPTS_PER_EXECUTION 100000

struct KernelString {
    char data[16];
    int len;
};

typedef struct {
    curandState* states;
} config;

// Prototypes
void vanity_setup(config& vanity, int gpu_index);
void vanity_run(config& vanity, int gpu_index, KernelString prefix, KernelString suffix);
void __global__ vanity_init(unsigned long long int* seed, curandState* state);
void __global__ vanity_scan(curandState* state, int* keys_found, int* gpu, int* execution_count, KernelString prefix, KernelString suffix);
bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz);

int main(int argc, char const* argv[]) {
    ed25519_set_verbose(false); // Less noise

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

    KernelString prefix_ks;
    prefix_ks.len = 0;
    if (prefix_str) {
        strncpy(prefix_ks.data, prefix_str, 15);
        prefix_ks.data[15] = '\0';
        prefix_ks.len = strlen(prefix_ks.data);
    }

    KernelString suffix_ks;
    suffix_ks.len = 0;
    if (suffix_str) {
        strncpy(suffix_ks.data, suffix_str, 15);
        suffix_ks.data[15] = '\0';
        suffix_ks.len = strlen(suffix_ks.data);
    }

    config vanity;
    vanity_setup(vanity, gpu_index);
    vanity_run(vanity, gpu_index, prefix_ks, suffix_ks);
    return 0;
}

std::string getTimeStr(){
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string s(30, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return s;
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
    cudaSetDevice(gpu_index);
    cudaDeviceProp device;
    cudaGetDeviceProperties(&device, gpu_index);

    int blockSize = 0, minGridSize = 0, maxActiveBlocks = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

    unsigned long long int rseed = makeSeed();
    unsigned long long int* dev_rseed;
    cudaMalloc((void**)&dev_rseed, sizeof(unsigned long long int));
    cudaMemcpy(dev_rseed, &rseed, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&(vanity.states), maxActiveBlocks * blockSize * sizeof(curandState));
    vanity_init<<<maxActiveBlocks, blockSize>>>(dev_rseed, vanity.states);
    cudaDeviceSynchronize();
}

void vanity_run(config &vanity, int gpu_index, KernelString prefix, KernelString suffix) {
    cudaSetDevice(gpu_index);

    int blockSize = 0, minGridSize = 0, maxActiveBlocks = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

    int keys_found_total = 0;
    int keys_found_this_iteration;
    int* dev_keys_found;
    int* dev_executions_this_gpu;
    int* dev_g;

    cudaMalloc((void**)&dev_keys_found, sizeof(int));
    cudaMalloc((void**)&dev_executions_this_gpu, sizeof(int));
    cudaMalloc((void**)&dev_g, sizeof(int));

    cudaMemcpy(dev_g, &gpu_index, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dev_keys_found, 0, sizeof(int));

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        cudaMemset(dev_executions_this_gpu, 0, sizeof(int));

        vanity_scan<<<maxActiveBlocks, blockSize>>>(vanity.states, dev_keys_found, dev_g, dev_executions_this_gpu, prefix, suffix);

        cudaDeviceSynchronize();

        cudaMemcpy(&keys_found_this_iteration, dev_keys_found, sizeof(int), cudaMemcpyDeviceToHost);
        keys_found_total = keys_found_this_iteration; // since atomicAdd accumulates

        if (keys_found_total >= STOP_AFTER_KEYS_FOUND) {
            exit(0);
        }
    }
}

void __global__ vanity_init(unsigned long long int* rseed, curandState* state) {
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    curand_init(*rseed + id, id, 0, &state[id]);
}

void __global__ vanity_scan(curandState* state, int* keys_found, int* gpu, int* exec_count, KernelString prefix, KernelString suffix) {
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    atomicAdd(exec_count, 1);

    // Local Kernel State
    ge_p3 A;
    curandState localState = state[id];
    unsigned char seed[32] = {0};
    unsigned char publick[32] = {0};
    unsigned char privatek[64] = {0};
    char key[256] = {0};

    // Initialize seed from state
    for (int i = 0; i < 32; ++i) {
        float random = curand_uniform(&localState);
        seed[i] = (uint8_t)(random * 255);
    }

    sha512_context md;

    for (int attempts = 0; attempts < ATTEMPTS_PER_EXECUTION; ++attempts) {
        // Optimized SHA512 (Inlined from original vanity.cu)
        md.curlen = 0; md.length = 0;
        md.state[0] = UINT64_C(0x6a09e667f3bcc908);
        md.state[1] = UINT64_C(0xbb67ae8584caa73b);
        md.state[2] = UINT64_C(0x3c6ef372fe94f82b);
        md.state[3] = UINT64_C(0xa54ff53a5f1d36f1);
        md.state[4] = UINT64_C(0x510e527fade682d1);
        md.state[5] = UINT64_C(0x9b05688c2b3e6c1f);
        md.state[6] = UINT64_C(0x1f83d9abfb41bd6b);
        md.state[7] = UINT64_C(0x5be0cd19137e2179);

        const unsigned char *in = seed;
        for (size_t i = 0; i < 32; i++) {
            md.buf[i + md.curlen] = in[i];
        }
        md.curlen += 32;

        md.length += md.curlen * UINT64_C(8);
        md.buf[md.curlen++] = (unsigned char)0x80;
        while (md.curlen < 120) {
            md.buf[md.curlen++] = (unsigned char)0;
        }
        STORE64H(md.length, md.buf+120);

        uint64_t S[8], W[80], t0, t1;
        int i;
        for (i = 0; i < 8; i++) S[i] = md.state[i];
        for (i = 0; i < 16; i++) LOAD64H(W[i], md.buf + (8*i));
        for (i = 16; i < 80; i++) W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];

        #define RND(a,b,c,d,e,f,g,h,i) \
        t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i]; \
        t1 = Sigma0(a) + Maj(a, b, c);\
        d += t0; \
        h  = t0 + t1;

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

        for (i = 0; i < 8; i++) md.state[i] = md.state[i] + S[i];
        for (i = 0; i < 8; i++) STORE64H(md.state[i], privatek+(8*i));

        privatek[0]  &= 248;
        privatek[31] &= 63;
        privatek[31] |= 64;

        ge_scalarmult_base(&A, privatek);
        ge_p3_tobytes(publick, &A);

        size_t keysize = 256;
        b58enc(key, &keysize, publick, 32);
        // b58enc sets null terminator. keysize includes null.

        // --- Matching Logic ---
        bool match = true;

        // Check Prefix
        if (prefix.len > 0) {
            for (int j = 0; j < prefix.len; ++j) {
                if (key[j] != prefix.data[j]) {
                    match = false;
                    break;
                }
            }
        }

        // Check Suffix
        if (match && suffix.len > 0) {
            int len = (int)keysize - 1;
            if (len < suffix.len) {
                match = false;
            } else {
                for (int j = 0; j < suffix.len; ++j) {
                    if (key[len - suffix.len + j] != suffix.data[j]) {
                        match = false;
                        break;
                    }
                }
            }
        }

        if (match) {
            atomicAdd(keys_found, 1);

            // Output JSON format
            printf("{\"public_key\": \"%s\", \"secret_key\": [", key);
            for(int n=0; n<32; n++) {
                printf("%d,", (unsigned char)seed[n]);
            }
            for(int n=0; n<32; n++) {
                printf("%d%s", publick[n], (n==31 ? "" : ","));
            }
            printf("]}\n");

            // Break inner loop to ensure we stop
            break;
        }

        // Increment Seed
        for (int i = 0; i < 32; ++i) {
            if (seed[i] == 255) {
                seed[i] = 0;
            } else {
                seed[i] += 1;
                break;
            }
        }
    }

    state[id] = localState;
}

bool __device__ b58enc(char *b58, size_t *b58sz, uint8_t *data, size_t binsz) {
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
