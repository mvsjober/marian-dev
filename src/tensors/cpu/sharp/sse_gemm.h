// Copyright (c) 2017 Microsoft Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "tensors/tensor.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"

#include <cassert>
#include <emmintrin.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

// This is a reference implementation of 16-bit matrix multiplication described in "Sharp Models on Dull Hardware: Fast and Accurate Neural Machine Translation Decoding on the CPU".
// This model is not as fast as the one in the paper, becuase it uses SSE2 instead of AVX2. AVX2 instructions are only available on more modern CPUs (Haswell or later).
// The only difference between SSE2 and AVX2 is that SSE operates on 128-bit vectors and AVX2 operates on 256-bit vecetors. So AVX2 can fit 16 16-bit integers intead of 8 8-bit integers.
// The algorithm is the same, you just replace these instructions with their 256-bit counterpart, i.e., _mm256_add_epi32, _mm256_madd_epi16, _mm256_hadd_epi32, ...
// Additional improvements can also be made from unrolling the for loop over num_B_rows in SSE_MatrixMult, which is not done here for clarity.

// ***************************************
// ************** IMPORTANT **************
// ***************************************
// The biggest "gotcha" when using this type of multiplication is dealing with overflow related to quantization.
// It is NOT enough to simply ensure that A and B fit into 16 bit integers. If A and B are quantized with $n$ bits,
// the result of multiplying them together will be quantized to $n^2$ bits. So if they are near the boundary of the 16-bit
// mark, then the result will be near 32-bits and overflow. However, if we use, say, n = 10 bits, then the product is 20 bits.
// This gives us 12 bits left over for the accumulation. So as long as the width of the common dimension is less than 2^12 = 4096, it is
// *impossible* to overflow. If we used, say, n = 12 bits, then we have 32-(12*2) = 8 bits left over. So we *could* overflow if width > 2^8.
//
// So, the tradeoff is between quantization precision and possibility of overflow. A good general value is 10 bits, since this gives high precision
// (precision is 1/2^10 ~= 0.001, which is more than what's needed for almost all neural nets), and cannot overflow unless the matrix width is > 4096.

// This quantizes floating point values into fixed-point 16-bit integers. Effectively, we are performing an SSE version of
// float x = ...;
// int16_t y = (int16_t)(quant_mult*x);
//
// Except that the casting is saturated. However, you should always ensure that the input fits into a fixed range anyways.
// I.e., you should ensure that quant_mult*x fits into the range [-2^15, 2^15].
// This should always be possible because the value you're quantizing will either be NN weights or NN activations, both of
// which can be clipped to a fixed range during training.

namespace marian {

namespace cpu {
namespace int16 {

const int BITS = 10;

static inline void Quantize(marian::Tensor out,
                            const marian::Tensor in) {
    int size = in->shape().elements();
    const float* input = in->data();
    __m256i* output = out->data<__m256i>();
    ABORT_IF(size % 16 != 0, "Size {} is not divisble by 8", size);
    ABORT_IF(reinterpret_cast<uintptr_t>(input) % 64 != 0, "Input {} is not 64-byte aligned", reinterpret_cast<uintptr_t>(input));
    ABORT_IF(reinterpret_cast<uintptr_t>(output) % 32 != 0, "Output {} is not 32-byte aligned", reinterpret_cast<uintptr_t>(output)); 

    float quant_mult = pow(2.0, (float)BITS);
    // Fill with the quantization multiplier.
    const __m512 quant_mult_reg = _mm512_set1_ps(quant_mult);
    const float *end = input + size;

    for (; input != end; input += 16, output += 1) {
      // Load 16 floats
      __m512 val = _mm512_load_ps(input);
      // Multiply each by the quantization factor.
      val = _mm512_mul_ps(val, quant_mult_reg);
      // Cast to 32-bit int
      __m512i as_int =  _mm512_cvtps_epi32(val);
      // Pack into 16-bit ints with saturation.
      // I would do two AVX512 registers and _mm512_packs_epi32 but that's not
      // AVX515F.
      *output = _mm256_packs_epi32(_mm512_castsi512_si256(as_int), _mm512_extracti64x4_epi64(as_int, 1));
    }
}

// Assuming sum1, sum2, sum3, and sum4 are arrays 32-bit signed integers,
// reduce within each.
// Returns [sum(sum1), sum(sum2), sum(sum3), sum(sum4)]
// TODO: consider doing in 64-bit, allowing 4 more bits of quantization?
inline __m128i Reduce(__m512i sum1, __m512i sum2, __m512i sum3, __m512i sum4) {
  // 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2
  __m512i pack12 = _mm512_add_epi32(_mm512_unpackhi_epi32(sum1, sum2), _mm512_unpacklo_epi32(sum1, sum2));
  // 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4
  __m512i pack34 = _mm512_add_epi32(_mm512_unpackhi_epi32(sum3, sum4), _mm512_unpacklo_epi32(sum3, sum4));
  // 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4
  __m512i pack1234 = _mm512_add_epi32(_mm512_unpackhi_epi64(pack12, pack34), _mm512_unpacklo_epi64(pack12, pack34));
  // Cut the register into halves and sum those.  1 2 3 4 1 2 3 4
  __m256i halves = _mm256_add_epi32(_mm512_castsi512_si256(pack1234), _mm512_extracti64x4_epi64(pack1234, 1));
  // Again: cut the register into halves and sum those. 1 2 3 4
  __m128i ret = _mm_add_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1));
  return ret;
}

union FloatAccess {
  float as_f[4];
  __m128 as_n;
};
union IntAccess {
  int32_t as_i[4];
  __m128i as_n;
};

static inline void AVX_MatrixMult(const __m512i * A, const __m512i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width) {
    ABORT_IF(width % 32, "Width {} is not a multiple of 32", width);
    ABORT_IF(reinterpret_cast<uintptr_t>(A) % 64, "A base pointer is not a multiple of 64");
    ABORT_IF(reinterpret_cast<uintptr_t>(B) % 64, "B base pointer is not a multiple of 64");
    const __m128 unquant_mult_sse = _mm_set1_ps(unquant_mult);

    const int sse_width = width/32;

    // We do loop unrolling over A. This is *significantly* faster
    // since B can live in the registers. We are assuming that
    // A is a multiple of 4, but we can add extra code to handle values of 1, 2, 3.
    //
    // We could also do loop unrolling over B, which adds some additional speedup.
    // We don't do that for the sake of clarity.
    // 
    // There are other memory access patterns we could do, e.g., put B on the outer loop.
    // The justification is that A is typically small enough that it can live in L1 cache.
    // B is usually a larger weight matrix, so it might not be able to. However, we are using
    // each element of B four times while it's still in a register, so caching is not as important.

    // Round down to a multiple of 4.
    int num_unroll_rows = num_A_rows & ~3;
    for (int i = 0; i < num_unroll_rows; i += 4) {
        const __m512i * A1_row = A + (i+0)*sse_width;
        const __m512i * A2_row = A + (i+1)*sse_width;
        const __m512i * A3_row = A + (i+2)*sse_width;
        const __m512i * A4_row = A + (i+3)*sse_width;

        for (int j = 0; j < num_B_rows; j++) {
            const __m512i * B_row = B + j*sse_width;

            __m512i sum1 = _mm512_setzero_si512();
            __m512i sum2 = _mm512_setzero_si512();
            __m512i sum3 = _mm512_setzero_si512();
            __m512i sum4 = _mm512_setzero_si512();

            // This is just a simple dot product, unrolled four ways.
            for (int k = 0; k < sse_width; k++) {
                __m512i b = *(B_row + k);

                __m512i a1 = *(A1_row + k);
                __m512i a2 = *(A2_row + k);
                __m512i a3 = *(A3_row + k);
                __m512i a4 = *(A4_row + k);

                // madd_epi16 does multiply add on 8 16-bit integers and accumulates into a four 32-bit register.
                // E.g.,
                // a1 = [f1, f2, f3, f4, f5, f6, f7, h8] (16-bit ints)
                // b1 = [h1, h2, h3, h4, h5, h6, h7, h8] (16-bit ints)
                // result = [f1*h1 + f2*h2, f3*h3 + f4*h4, f5*h5 + f6*h6, f7*h7 + f8*h8] (32-bit ints)
                // Then add_epi32 just effectively does a += on these 32-bit integers.
                sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
                sum2 = _mm512_add_epi32(sum2, _mm512_madd_epi16(b, a2));
                sum3 = _mm512_add_epi32(sum3, _mm512_madd_epi16(b, a3));
                sum4 = _mm512_add_epi32(sum4, _mm512_madd_epi16(b, a4));
            }
            FloatAccess a;
            // Get floats for each of the sums to write.
            a.as_n = _mm_cvtepi32_ps(Reduce(sum1, sum2, sum3, sum4));
            // Undo quantization scaling.
            a.as_n = _mm_mul_ps(a.as_n, unquant_mult_sse);
            // Also note that the memory acceses on C are not consecutive, but this is a tradeoff that we have to make.
            // We can't have consecutive accesses of A, B, *and* C. But we access A and B a lot more so it makes
            // sense to do it this way.
            // Scatter to outputs:
            *(C + (i+0)*num_B_rows + j) = a.as_f[0];
            *(C + (i+1)*num_B_rows + j) = a.as_f[1];
            *(C + (i+2)*num_B_rows + j) = a.as_f[2];
            *(C + (i+3)*num_B_rows + j) = a.as_f[3];
            /* Sadly the scatter instruction requires avx512vl
             * _mm_i32scatter_ps(C + i * num_B_rows + j, num_b_rows_scatter, float_sums, sizeof(float));
             */
        }
    }
    // Handle the non-multiples of 4 rows.
    // TODO: efficient version for 3 rows, 2 rows, etc.
    for (int i = num_unroll_rows; i < num_A_rows; ++i) {
      const __m512i * A1_row = A + i * sse_width;
      for (int j = 0; j < num_B_rows; j++) {
        __m512i sum1 = _mm512_setzero_si512();
        for (int k = 0; k < sse_width; k++) {
          const __m512i * B_row = B + j*sse_width;
          __m512i b = *(B_row + k);
          __m512i a1 = *(A1_row + k);
          sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
        }
        // Fold register over itself.
        __m256i halves = _mm256_add_epi32(_mm512_castsi512_si256(sum1), _mm512_extracti64x4_epi64(sum1, 1));
        IntAccess a;
        a.as_n = _mm_add_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1));
        // TODO is there a more efficient way?
        *(C + (i)*num_B_rows + j) = unquant_mult * static_cast<float>(a.as_i[0] + a.as_i[1] + a.as_i[2] + a.as_i[3]);
      }
    }
}

static void AddBias(marian::Tensor C, const marian::Tensor Bias) {
    float* y = C->data();
    const float* x = C->data();
    const float* bias = Bias->data();

    int m = C->shape().elements() / C->shape()[-1];
    int n = C->shape()[-1];
    int n16 = n & ~15;
//    int n4 = n & ~3;

    for(int j = 0; j < m; ++j) {
        int i = 0;
        for (; i < n16; i += 16) {
            __m512 ai = _mm512_loadu_ps(x + j * n + i);
            __m512 bi = _mm512_loadu_ps(bias + i);
            __m512 yi = _mm512_add_ps(ai, bi);
            _mm512_storeu_ps(y + j * n + i, yi);
        }
        // Seems to slightly degrade performance.
/*        for (; i < n4; i += 4) {
            __m128 ai = _mm_loadu_ps(x + j * n + i);
            __m128 bi = _mm_loadu_ps(bias + i);
            __m128 yi = _mm_add_ps(ai, bi);
            _mm_storeu_ps(y + j * n + i, yi);
        }*/
        for (; i < n; i++) {
            y[j * n + i] = x[j * n + i] + bias[i];
        }
    }
}


static void ProdInt(marian::Tensor C,
                    const marian::Tensor A,
                    const marian::Tensor B,
                    float scale) {

    ABORT_IF(scale != 1, "Scale other than 1 not supported");

    // @TODO: make this a parameter
    float quant_mult = pow(2.0, (float)BITS);

    // If we quantize to n bits and then multiple the values together, the result will be quantized to n^2 bits.
    // So we must divide by 1.0/(n^2) to get back the original value.
    float unquant_mult = 1.0 / (quant_mult * quant_mult);

    float* fC = C->data();

    int num_A_rows = A->shape().elements() / A->shape()[-1];
    int num_B_rows = B->shape().elements() / B->shape()[-1];
    int width = B->shape()[-1];

    AVX_MatrixMult(A->data<__m512i>(), B->data<__m512i>(), fC, unquant_mult, num_A_rows, num_B_rows, width);
}

}
}

}
