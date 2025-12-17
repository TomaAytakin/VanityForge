/*
 * VanityForge - Phase 1: High-Throughput Iterator (Scalar+Point Add)
 * Target: NVIDIA L4 (SM89)
 *
 * Architecture:
 * - Phase 1 (GPU): Scalar Iterator + Point Addition (P' = P + B)
 *   - Avoids SHA-512 in hot loop.
 *   - Uses Fe/Ge optimized for registers.
 *   - Batched Inversion (Size 4) to amortize fe_invert cost.
 * - Ring Buffer: Transfers 'scalar index' to Host.
 * - Phase 2 (Host): Reconstructs Scalar, Point, and Base58 string.
 *
 * Optimization:
 * - <64 Registers per thread (Target)
 * - Zero spills
 * - Persistent Kernel
 * - Shared Memory for Batch Storage
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
#include "sha512.cu"

// --- Configuration ---

#define BLOCK_SIZE 128
#define BATCH_SIZE 4
#define RING_BUFFER_SIZE 1024
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

// --- Constant Memory ---

__constant__ ge_precomp dc_B;

// --- Kernel ---

__global__ __launch_bounds__(BLOCK_SIZE, 2)
void phase1_filter_kernel(
    uint64_t target_prefix,
    uint64_t target_mask,
    DeviceRingBuffer* ring,
    Stats* stats
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t total_threads = gridDim.x * blockDim.x;

    // Shared Memory for Batch Storage
    // Structure: [BATCH][BLOCK] for coalescing?
    // z[batch_idx * BLOCK + tid]
    __shared__ fe batch_z[BATCH_SIZE * BLOCK_SIZE];
    __shared__ fe batch_y[BATCH_SIZE * BLOCK_SIZE];

    // Initialize P = tid * B
    ge_p3 P;

    // Construct scalar for initialization: just 'tid'.
    // We start at offset 'tid' and stride 'total_threads' effectively?
    // No, we use Chunking.
    // Thread i handles [i*BATCH + k*Grid*BATCH, i*BATCH + (k*Grid+1)*BATCH) ?
    // Simpler:
    // P_init = (tid * BATCH_SIZE) * B
    // Then loop BATCH_SIZE times (P+=B).
    // Then Jump by (TotalThreads * BATCH_SIZE - (BATCH_SIZE-1)) * B? No.
    // Next start = (tid * BATCH) + (TotalThreads * BATCH).
    // Current End P = (tid * BATCH) + (BATCH-1).
    // Jump = (TotalThreads * BATCH) - (BATCH-1).
    //
    // Let's implement P_init = (tid * BATCH) * B.

    unsigned char init_scalar[32];
    #pragma unroll
    for(int i=0; i<32; i++) init_scalar[i] = 0;

    // tid * BATCH
    // We assume tid * BATCH fits in 64 bits (tid ~ 20k, BATCH=4 -> 80k << 2^64).
    uint64_t start_index = (uint64_t)tid * BATCH_SIZE;
    uint64_t t_temp = start_index;
    for(int i=0; i<8; i++) {
        init_scalar[i] = t_temp & 0xFF;
        t_temp >>= 8;
    }

    ge_scalarmult_base(&P, init_scalar);

    // Precompute Big Step Point
    // Step = (TotalThreads * BATCH - (BATCH - 1)) * B
    // Wait, simpler.
    // Jump from Start of Batch to Start of Next Batch.
    // Jump = TotalThreads * BATCH * B.
    // But we are at End of Batch inside loop?
    // No, we can preserve Start P if we want?
    // No, updating P is faster.
    // If P is at (Start + BATCH - 1).
    // Next Start is (Start + TotalThreads * BATCH).
    // Delta = TotalThreads * BATCH - (BATCH - 1).

    ge_p3 P_jump;
    unsigned char step_scalar[32];
    #pragma unroll
    for(int i=0; i<32; i++) step_scalar[i] = 0;

    t_temp = (uint64_t)total_threads * BATCH_SIZE - (BATCH_SIZE - 1);
    for(int i=0; i<8; i++) {
        step_scalar[i] = t_temp & 0xFF;
        t_temp >>= 8;
    }
    ge_scalarmult_base(&P_jump, step_scalar);

    ge_cached P_jump_cached;
    ge_p3_to_cached(&P_jump_cached, &P_jump);

    uint64_t global_iter = 0;

    while (true) {
        // 1. Generate Batch
        #pragma unroll
        for (int i = 0; i < BATCH_SIZE; i++) {
            // Save Z and Y
            // batch_z[i][tid]
            int idx = i * BLOCK_SIZE + threadIdx.x;
            fe_copy(batch_z[idx], P.Z);
            fe_copy(batch_y[idx], P.Y);

            // If not last, Step P += B
            if (i < BATCH_SIZE - 1) {
                ge_p1p1 next_P;
                ge_madd(&next_P, &P, &dc_B);
                ge_p1p1_to_p3(&P, &next_P);
            }
        }

        // 2. Batch Inversion
        // Accumulate
        fe product;
        fe_1(product);

        // We need to store intermediates to back-substitute.
        // We can't store them all in registers if BATCH is large.
        // BATCH=4. 4 * 40 = 160 bytes. Fits in registers.
        fe intermediates[BATCH_SIZE];

        #pragma unroll
        for (int i = 0; i < BATCH_SIZE; i++) {
            int idx = i * BLOCK_SIZE + threadIdx.x;
            fe_copy(intermediates[i], product);
            fe_mul(product, product, batch_z[idx]);
        }

        // Invert total product
        fe inv_product;
        fe_invert(inv_product, product);

        // Back-substitute to get individual inverses
        #pragma unroll
        for (int i = BATCH_SIZE - 1; i >= 0; i--) {
            int idx = i * BLOCK_SIZE + threadIdx.x;
            fe inv_z; // 1/z[i]
            fe_mul(inv_z, inv_product, intermediates[i]); // inv * prod_prev

            // Update inv_product for next
            fe_mul(inv_product, inv_product, batch_z[idx]);

            // 3. Check
            // y_affine = Y * inv_z
            fe y_aff;
            fe_mul(y_aff, batch_y[idx], inv_z);

            unsigned char s[32];
            fe_tobytes(s, y_aff);

            // Filter
            uint64_t key_prefix =
                ((uint64_t)s[0] << 56) | ((uint64_t)s[1] << 48) |
                ((uint64_t)s[2] << 40) | ((uint64_t)s[3] << 32) |
                ((uint64_t)s[4] << 24) | ((uint64_t)s[5] << 16) |
                ((uint64_t)s[6] << 8)  | ((uint64_t)s[7]);

            if ((key_prefix & target_mask) == target_prefix) {
                uint32_t slot = atomicAdd(&ring->write_head, 1);
                Candidate c;
                c.thread_id = tid;
                c.iter_count = global_iter * BATCH_SIZE + i;
                ring->items[slot & RING_BUFFER_MASK] = c;
            }
        }

        // 4. Big Step
        ge_p1p1 next_P;
        ge_add(&next_P, &P, &P_jump_cached);
        ge_p3_to_p3(&P, &next_P);

        global_iter++;

        if (threadIdx.x == 0) {
            atomicAdd((unsigned long long*)&stats->total_hashes, (unsigned long long)(BATCH_SIZE * blockDim.x));
        }
    }
}

// --- Host Logic: Phase 2 ---

// Solve: reconstruct key from Ring Buffer item
void phase2_solve(uint64_t thread_id, uint64_t iter_count, uint32_t total_threads, const char* suffix_check) {
    // Reconstruct Scalar
    // The iteration logic in kernel:
    // P_tid starts at `tid * BATCH`.
    // Each loop step (global_iter) advances by `total_threads * BATCH`.
    // Inside loop, we check `global_iter * BATCH + i`.
    // So the absolute index is:
    // Index = (tid * BATCH) + (global_iter * TotalThreads * BATCH) + i?
    // Wait.
    // Iter 0: tid*BATCH + 0, tid*BATCH + 1 ...
    // Iter 1: tid*BATCH + TotalThreads*BATCH + 0 ...
    //
    // So Index = (tid * BATCH) + iter_count * (TotalThreads * BATCH / BATCH) ?
    // `iter_count` passed from kernel is `global_iter * BATCH + i`.
    // Let K = iter_count.
    // BatchIndex = K / BATCH = global_iter.
    // Offset = K % BATCH = i.
    // Index = (tid * BATCH) + BatchIndex * (TotalThreads * BATCH) + Offset.
    //       = BATCH * (tid + BatchIndex * TotalThreads) + Offset.

    // Let's verify.
    // Scalar = Index.

    uint64_t k = iter_count;
    uint64_t batch_idx = k / BATCH_SIZE;
    uint64_t offset = k % BATCH_SIZE;

    unsigned __int128 scalar_idx = (unsigned __int128)BATCH_SIZE * (thread_id + (unsigned __int128)batch_idx * total_threads) + offset;

    unsigned char scalar[32];
    memset(scalar, 0, 32);

    unsigned __int128 temp = scalar_idx;
    for(int i=0; i<32; i++) { // Fill up to 32 bytes (128 bit int fits in 16 bytes)
        if (i < 16) scalar[i] = (unsigned char)(temp & 0xFF);
        else scalar[i] = 0;
        temp >>= 8;
    }

    ge_p3 A;
    ge_scalarmult_base(&A, scalar);

    unsigned char publick[32];
    ge_p3_tobytes(publick, &A);

    char b58[128];
    size_t b58len = 128;
    b58enc(b58, &b58len, publick, 32);

    if (suffix_check && strlen(suffix_check) > 0) {
        size_t len = strlen(b58);
        size_t slen = strlen(suffix_check);
        if (len >= slen && strcmp(b58 + len - slen, suffix_check) == 0) {
             printf("{\"found\": true, \"public_key\": \"%s\", \"secret_key\": [", b58);
             for(int i=0; i<32; i++) printf("%d, ", scalar[i]);
             for(int i=0; i<31; i++) printf("%d, ", publick[i]);
             printf("%d]}\n", publick[31]);
             fflush(stdout);
        }
    } else {
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
    bool prefix_set = false;
    bool mask_set = false;

    // Parsing
    for(int i=1; i<argc; i++) {
        if (strcmp(argv[i], "--suffix")==0 && i+1<argc) suffix = argv[i+1];
        if (strcmp(argv[i], "--device")==0 && i+1<argc) device = atoi(argv[i+1]);
        if (strcmp(argv[i], "--prefix-val")==0 && i+1<argc) {
            prefix_val = strtoull(argv[i+1], NULL, 16);
            prefix_set = true;
        }
        if (strcmp(argv[i], "--mask-val")==0 && i+1<argc) {
            mask_val = strtoull(argv[i+1], NULL, 16);
            mask_set = true;
        }
        if (strcmp(argv[i], "--gpu-index")==0 && i+1<argc) device = atoi(argv[i+1]);
    }

    if (!prefix_set || !mask_set || mask_val == 0) {
        fprintf(stderr, "Error: --prefix-val and --mask-val (non-zero) are required.\n");
        return 1;
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

    // Copy Base Point to Constant Memory
    // base[0][0] from precomp_data.h (included via ge.cu)
    cudaMemcpyToSymbol(dc_B, &base[0][0], sizeof(ge_precomp));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int gridSize = prop.multiProcessorCount * 256; // High occupancy

    printf("VanityForge Iterator (L4 Optimized)\n");
    printf("GPU: %s\n", prop.name);
    printf("Config: Block=%d, Batch=%d, Grid=%d\n", BLOCK_SIZE, BATCH_SIZE, gridSize);
    printf("Target Prefix: %lx Mask: %lx\n", prefix_val, mask_val);

    phase1_filter_kernel<<<gridSize, BLOCK_SIZE>>>(
        prefix_val,
        mask_val,
        d_ring,
        d_stats
    );

    CHECK_CUDA(cudaGetLastError());

    uint32_t local_read_head = 0;
    unsigned long long last_hashes = 0;
    uint32_t total_threads = gridSize * BLOCK_SIZE;

    while(1) {
        usleep(200000);

        uint32_t write_head = h_ring_mapped->write_head;

        while (local_read_head < write_head) {
            Candidate c = h_ring_mapped->items[local_read_head & RING_BUFFER_MASK];
            phase2_solve(c.thread_id, c.iter_count, total_threads, suffix);
            local_read_head++;
        }

        unsigned long long current = h_stats_mapped->total_hashes;
        double speed = (double)(current - last_hashes) * 5.0 / 1000000.0;
        last_hashes = current;

        static int ticks = 0;
        if (ticks++ % 5 == 0) {
            fprintf(stderr, "[Status] Speed: %.2f MH/s\n", speed);
        }
    }

    return 0;
}
