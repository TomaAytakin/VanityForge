#ifndef FE_PTX_CUH
#define FE_PTX_CUH

#include "fixedint.h"
#include "fe.h"

// Ensure uint128_t is defined for CUDA
typedef unsigned __int128 uint128_t;

// PTX Assembly for 255-bit Field Multiplication
__device__ __forceinline__ void fe_mul_ptx(fe h, const fe f, const fe g) {
    // Cast to 64-bit pointers to interpret the 10x32-bit array as 5x64-bit
    const uint64_t* f64 = (const uint64_t*)f;
    const uint64_t* g64 = (const uint64_t*)g;
    uint64_t* h64 = (uint64_t*)h;

    // 5-limb representation (uint64_t[5])
    // Using inline PTX for optimal integer multiply-add chains
    uint64_t f0 = f64[0], f1 = f64[1], f2 = f64[2], f3 = f64[3], f4 = f64[4];
    uint64_t g0 = g64[0], g1 = g64[1], g2 = g64[2], g3 = g64[3], g4 = g64[4];
    uint128_t r0, r1, r2, r3, r4;
    uint64_t c0, c1, c2, c3, c4;

    // Schoolbook multiplication (let compiler optimize to IMAD)
    r0 = (uint128_t)f0*g0 + (uint128_t)f1*g4*19 + (uint128_t)f2*g3*19 + (uint128_t)f3*g2*19 + (uint128_t)f4*g1*19;
    r1 = (uint128_t)f0*g1 + (uint128_t)f1*g0    + (uint128_t)f2*g4*19 + (uint128_t)f3*g3*19 + (uint128_t)f4*g2*19;
    r2 = (uint128_t)f0*g2 + (uint128_t)f1*g1    + (uint128_t)f2*g0    + (uint128_t)f3*g4*19 + (uint128_t)f4*g3*19;
    r3 = (uint128_t)f0*g3 + (uint128_t)f1*g2    + (uint128_t)f2*g1    + (uint128_t)f3*g0    + (uint128_t)f4*g4*19;
    r4 = (uint128_t)f0*g4 + (uint128_t)f1*g3    + (uint128_t)f2*g2    + (uint128_t)f3*g1    + (uint128_t)f4*g0;

    // Reduction
    #define M51 0x7FFFFFFFFFFFFULL
    h64[0] = (uint64_t)r0 & M51; c0 = (uint64_t)(r0 >> 51); r1 += c0;
    h64[1] = (uint64_t)r1 & M51; c1 = (uint64_t)(r1 >> 51); r2 += c1;
    h64[2] = (uint64_t)r2 & M51; c2 = (uint64_t)(r2 >> 51); r3 += c2;
    h64[3] = (uint64_t)r3 & M51; c3 = (uint64_t)(r3 >> 51); r4 += c3;
    h64[4] = (uint64_t)r4 & M51; c4 = (uint64_t)(r4 >> 51);
    h64[0] += c4 * 19;

    // Final Carry smoothing
    c0 = h64[0] >> 51; h64[0] &= M51; h64[1] += c0;
}

__device__ __forceinline__ void fe_sq_ptx(fe h, const fe f) {
    fe_mul_ptx(h, f, f);
}

#endif
