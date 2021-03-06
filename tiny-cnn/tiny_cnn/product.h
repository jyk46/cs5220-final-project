/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#if defined(CNN_USE_SSE) || defined(CNN_USE_AVX) || defined(CNN_USE_AVX512)
#include <immintrin.h>
#endif
#include <cstdint>
#include <cassert>
#include <numeric>

#if defined(_MSC_VER)
#define VECTORIZE_ALIGN(x) __declspec(align(x))
#elif defined(__GNUC__)
#define VECTORIZE_ALIGN(x) __attribute__((aligned(x)))
#else
#define VECTORIZE_ALIGN(x) __attribute__((aligned(x)))
#endif

namespace vectorize {
namespace detail {


template<typename T>
inline bool is_aligned(T, const typename T::value_type* p) {
    return is_aligned(T(), p);
}

template<typename T>
inline bool is_aligned(T, const typename T::value_type* p1, const typename T::value_type* p2) {
    return is_aligned(T(), p1) && is_aligned(T(), p2);
}

// traits

template <typename T>
struct generic {
    typedef T register_type;
    typedef T value_type;
    enum {
        unroll_size = 1
    };
    static register_type set1(const value_type& x) { return x; }
    static register_type zero() { return 0.0; }
    static register_type mul(const register_type& v1, const register_type& v2) { return v1 * v2; }
    static register_type add(const register_type& v1, const register_type& v2) { return v1 + v2; }
    static register_type load(const value_type* px) { return *px; }
    static register_type loadu(const value_type* px) { return *px; }
    static void store(value_type* px, const register_type& v) { *px = v; }
    static void storeu(value_type* px, const register_type& v) { *px = v; }
    static value_type resemble(const register_type& x) { return x; }
};

#ifdef CNN_USE_SSE

struct float_sse {
    typedef __m128 register_type;
    typedef float value_type;
    enum {
        unroll_size = 4
    };
    static register_type set1(const value_type& x) { return _mm_set1_ps(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm_mul_ps(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm_add_ps(v1, v2); }
    static register_type load(const value_type* px) { return _mm_load_ps(px); }
    static register_type loadu(const value_type* px) { return _mm_loadu_ps(px); }
    static void store(value_type* px, const register_type& v) { _mm_store_ps(px, v); }
    static void storeu(value_type* px, const register_type& v) { _mm_storeu_ps(px, v); }
    static value_type resemble(const register_type& x) {
        VECTORIZE_ALIGN(16) float tmp[4];
        _mm_store_ps(tmp, x);
        return tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
};

struct double_sse {
    typedef __m128d register_type;
    typedef double value_type;
    enum {
        unroll_size = 2
    };
    static register_type set1(const value_type& x) { return _mm_set1_pd(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm_mul_pd(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm_add_pd(v1, v2); }
    static register_type load(const value_type* px) { return _mm_load_pd(px); }
    static register_type loadu(const value_type* px) { return _mm_loadu_pd(px); }
    static void store(value_type* px, const register_type& v) { _mm_store_pd(px, v); }
    static void storeu(value_type* px, const register_type& v) { _mm_storeu_pd(px, v); }
    static value_type resemble(const register_type& x) {
        VECTORIZE_ALIGN(16) double tmp[2];
        _mm_store_pd(tmp, x);
        return tmp[0] + tmp[1];
    }
};

template<typename T>
struct sse {};
template<>
struct sse<float> : public float_sse {};
template<>
struct sse<double> : public double_sse {};

template<typename T>
inline bool is_aligned(sse<T>, const typename sse<T>::value_type* p) {
    return reinterpret_cast<size_t>(p) % 16 == 0;
}

#endif // CNN_USE_SSE

#ifdef CNN_USE_AVX512

struct float_avx512 {
    typedef __m512 register_type;
    typedef float value_type;
    enum {
        unroll_size = 16
    };
    static register_type set(const value_type& v0, const value_type& v1,
                             const value_type& v2, const value_type& v3,
                             const value_type& v4, const value_type& v5,
                             const value_type& v6, const value_type& v7,
                             const value_type& v8, const value_type& v9,
                             const value_type& v10, const value_type& v11,
                             const value_type& v12, const value_type& v13,
                             const value_type& v14, const value_type& v15)
    {
      return _mm512_set_ps(v15, v14, v13, v12, v11, v10, v9, v8,
                           v7, v6, v5, v4, v3, v2, v1, v0);
    }
    static register_type set1(const value_type& x) { return _mm512_set1_ps(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm512_mul_ps(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm512_add_ps(v1, v2); }
    static register_type load(const value_type* px) { return _mm512_load_ps(px); }
//    static register_type loadu(const value_type* px) { return _mm512_loadu_ps(px); }
    static void store(value_type* px, const register_type& v) { _mm512_store_ps(px, v); }
//    static void storeu(value_type* px, const register_type& v) { _mm512_storeu_ps(px, v); }
    static value_type resemble(const register_type& x) {
        VECTORIZE_ALIGN(64) float tmp[16];
        _mm512_store_ps(tmp, x);
        return std::accumulate(tmp, tmp + 16, 0.0f);
    }
};

struct double_avx512 {
    typedef __m512d register_type;
    typedef double value_type;
    enum {
        unroll_size = 8
    };
    static register_type set(const value_type& v0, const value_type& v1,
                             const value_type& v2, const value_type& v3,
                             const value_type& v4, const value_type& v5,
                             const value_type& v6, const value_type& v7)
    {
      return _mm512_set_pd(v7, v6, v5, v4, v3, v2, v1, v0);
    }
    static register_type set1(const value_type& x) { return _mm512_set1_pd(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm512_mul_pd(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm512_add_pd(v1, v2); }
    static register_type load(const value_type* px) { return _mm512_load_pd(px); }
//    static register_type loadu(const value_type* px) { return _mm512_loadu_pd(px); }
    static void store(value_type* px, const register_type& v) { _mm512_store_pd(px, v); }
//    static void storeu(value_type* px, const register_type& v) { _mm512_storeu_pd(px, v); }
    static value_type resemble(const register_type& x) {
        VECTORIZE_ALIGN(64) double tmp[8];
        _mm512_store_pd(tmp, x);
        return std::accumulate(tmp, tmp + 8, 0.0);
    }
};

template<typename T>
struct avx512 {};
template<>
struct avx512<float> : public float_avx512 {};
template<>
struct avx512<double> : public double_avx512 {};

template<typename T>
inline bool is_aligned(avx512<T>, const typename avx512<T>::value_type* p) {
    return reinterpret_cast<size_t>(p) % 64 == 0;
}

#elif defined(CNN_USE_AVX)

struct float_avx {
    typedef __m256 register_type;
    typedef __m256i mask_type;
    typedef float value_type;
    enum {
        unroll_size = 8
    };
    static register_type set1(const value_type& x) { return _mm256_set1_ps(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm256_mul_ps(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm256_add_ps(v1, v2); }
    static register_type load(const value_type* px) { return _mm256_load_ps(px); }
    static register_type loadu(const value_type* px) { return _mm256_loadu_ps(px); }
    static void store(value_type* px, const register_type& v) { _mm256_store_ps(px, v); }
    static void storeu(value_type* px, const register_type& v) { _mm256_storeu_ps(px, v); }
    static value_type resemble(const register_type& x) {
        VECTORIZE_ALIGN(32) float tmp[8];
        _mm256_store_ps(tmp, x);
        return std::accumulate(tmp, tmp + 8, 0.0f);
    }

    static register_type setzero() { return _mm256_setzero_ps(); }

    static register_type fmadd(
        const register_type& v1,
        const register_type& v2,
        const register_type& v3) {
      return _mm256_fmadd_ps(v1, v2, v3);
    }

    static register_type maskload(const value_type* px, const mask_type& m) {
      return _mm256_maskload_ps(px, m);
    }

    static void maskstore(
        value_type* px,
        const mask_type& m,
        const register_type& v) {
      _mm256_maskstore_ps(px, m, v);
    }

    // Helper for calculating mask for vector loads/stores. Enabled mask
    // values need to have the MSB set in order for the masked vector
    // load/stores to detect it correctly so we use -1 instead of 1.
    static mask_type generate_mask(int num_valid) {
      char mask  = 0xff;
      int  shamt = unroll_size - num_valid;
      mask >>= shamt;

      int MASK_1 = 0xffffffff;
      int MASK_0 = 0x00000000;

      return _mm256_set_epi32(
          ((mask >> 7) & 0x1) ? MASK_1 : MASK_0,
          ((mask >> 6) & 0x1) ? MASK_1 : MASK_0,
          ((mask >> 5) & 0x1) ? MASK_1 : MASK_0,
          ((mask >> 4) & 0x1) ? MASK_1 : MASK_0,
          ((mask >> 3) & 0x1) ? MASK_1 : MASK_0,
          ((mask >> 2) & 0x1) ? MASK_1 : MASK_0,
          ((mask >> 1) & 0x1) ? MASK_1 : MASK_0,
          ((mask >> 0) & 0x1) ? MASK_1 : MASK_0
      );
    }
};

struct double_avx {
    typedef __m256d register_type;
    typedef __m256i mask_type;
    typedef double value_type;
    enum {
        unroll_size = 4
    };
    static register_type set1(const value_type& x) { return _mm256_set1_pd(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm256_mul_pd(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm256_add_pd(v1, v2); }
    static register_type load(const value_type* px) { return _mm256_load_pd(px); }
    static register_type loadu(const value_type* px) { return _mm256_loadu_pd(px); }
    static void store(value_type* px, const register_type& v) { _mm256_store_pd(px, v); }
    static void storeu(value_type* px, const register_type& v) { _mm256_storeu_pd(px, v); }
    static value_type resemble(const register_type& x) {
        VECTORIZE_ALIGN(32) double tmp[4];
        _mm256_store_pd(tmp, x);
        return std::accumulate(tmp, tmp + 4, 0.0);
    }

    static register_type setzero() { return _mm256_setzero_pd(); }

    static register_type fmadd(
        const register_type& v1,
        const register_type& v2,
        const register_type& v3) {
      return _mm256_fmadd_pd(v1, v2, v3);
    }

    static register_type maskload(const value_type* px, const mask_type& m) {
      return _mm256_maskload_pd(px, m);
    }

    static void maskstore(
        value_type* px,
        const mask_type& m,
        const register_type& v) {
      _mm256_maskstore_pd(px, m, v);
    }

    // Helper for calculating mask for vector loads/stores. Enabled mask
    // values need to have the MSB set in order for the masked vector
    // load/stores to detect it correctly so we use -1 instead of 1.
    static mask_type generate_mask(int num_valid) {
      char mask  = 0x0f;
      int  shamt = unroll_size - num_valid;
      mask >>= shamt;

      __int64_t MASK_1 = 0xffffffffffffffff;
      __int64_t MASK_0 = 0x0000000000000000;

      return _mm256_set_epi64x(
          ((mask >> 3) & 0x1) ? MASK_1 : MASK_0,
          ((mask >> 2) & 0x1) ? MASK_1 : MASK_0,
          ((mask >> 1) & 0x1) ? MASK_1 : MASK_0,
          ((mask >> 0) & 0x1) ? MASK_1 : MASK_0
      );
    }
};

template<typename T>
struct avx {};
template<>
struct avx<float> : public float_avx {};
template<>
struct avx<double> : public double_avx {};

template<typename T>
inline bool is_aligned(avx<T>, const typename avx<T>::value_type* p) {
    return reinterpret_cast<size_t>(p) % 32 == 0;
}

#endif // CNN_USE_AVX

// generic dot-product
template<typename T>
inline typename T::value_type dot_product_nonaligned(const typename T::value_type* f1, const typename T::value_type* f2, unsigned int size) {
    typename T::register_type result = T::zero();

    for (unsigned int i = 0; i < size/T::unroll_size; i++) {
    // The version of ICC in the course does not support unaligned
    // vector memops for AVX-512, so we need to use scalar loads to fill
    // the vector registers for the multiply. This assumes operation on
    // doubles only!
    #ifdef CNN_USE_AVX512
        typename T::register_type f1_vec = _mm512_set_pd(
            f1[i*T::unroll_size+7],
            f1[i*T::unroll_size+6],
            f1[i*T::unroll_size+5],
            f1[i*T::unroll_size+4],
            f1[i*T::unroll_size+3],
            f1[i*T::unroll_size+2],
            f1[i*T::unroll_size+1],
            f1[i*T::unroll_size+0]
        );

        typename T::register_type f2_vec = _mm512_set_pd(
            f2[i*T::unroll_size+7],
            f2[i*T::unroll_size+6],
            f2[i*T::unroll_size+5],
            f2[i*T::unroll_size+4],
            f2[i*T::unroll_size+3],
            f2[i*T::unroll_size+2],
            f2[i*T::unroll_size+1],
            f2[i*T::unroll_size+0]
        );

        result = T::add(result, T::mul(f1_vec, f2_vec));
    #else
        result = T::add(result, T::mul(T::loadu(&f1[i*T::unroll_size]), T::loadu(&f2[i*T::unroll_size])));
    #endif
    }

    typename T::value_type sum = T::resemble(result);

    for (unsigned int i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        sum += f1[i] * f2[i];

    return sum;
}

// generic dot-product(aligned)
template<typename T>
inline typename T::value_type dot_product_aligned(const typename T::value_type* f1, const typename T::value_type* f2, unsigned int size) {
    typename T::register_type result = T::zero();

    assert(is_aligned(T(), f1));
    assert(is_aligned(T(), f2));

    for (unsigned int i = 0; i < size/T::unroll_size; i++)
        result = T::add(result, T::mul(T::load(&f1[i*T::unroll_size]), T::load(&f2[i*T::unroll_size])));

    typename T::value_type sum = T::resemble(result);

    for (unsigned int i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        sum += f1[i] * f2[i];

    return sum;
}

template<typename T>
inline void muladd_aligned(const typename T::value_type* src, typename T::value_type c, unsigned int size, typename T::value_type* dst) {
    typename T::register_type factor = T::set1(c);

    for (unsigned int i = 0; i < size/T::unroll_size; i++) {
        typename T::register_type d = T::load(&dst[i*T::unroll_size]);
        typename T::register_type s = T::load(&src[i*T::unroll_size]);
        T::store(&dst[i*T::unroll_size], T::add(d, T::mul(s, factor)));
    }

    for (unsigned int i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        dst[i] += src[i] * c;
}


template<typename T>
inline void muladd_nonaligned(const typename T::value_type* src, typename T::value_type c, unsigned int size, typename T::value_type* dst) {

    // The version of ICC in the course does not support unaligned
    // vector memops for AVX-512, so we need to use scalar loads to fill
    // the vector registers for the multiply. This assumes operation on
    // doubles only!

    #ifdef CNN_USE_AVX512
    for (unsigned int i = 0; i < size; i++)
        dst[i] += src[i] * c;
    #else

    typename T::register_type factor = T::set1(c);

    for (unsigned int i = 0; i < size/T::unroll_size; i++) {
        typename T::register_type d = T::loadu(&dst[i*T::unroll_size]);
        typename T::register_type s = T::loadu(&src[i*T::unroll_size]);
        T::storeu(&dst[i*T::unroll_size], T::add(d, T::mul(s, factor)));
    }

    for (unsigned int i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        dst[i] += src[i] * c;

    #endif
}

template<typename T>
inline void reduce_aligned(const typename T::value_type* src, unsigned int size, typename T::value_type* dst) {
    for (unsigned int i = 0; i < size/T::unroll_size; i++) {
        typename T::register_type d = T::load(&dst[i*T::unroll_size]);
        typename T::register_type s = T::load(&src[i*T::unroll_size]);
        T::store(&dst[i*T::unroll_size], T::add(d, s));
    }

    for (unsigned int i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        dst[i] += src[i];
}

template<typename T>
inline void reduce_nonaligned(const typename T::value_type* src, unsigned int size, typename T::value_type* dst) {

    // The version of ICC in the course does not support unaligned
    // vector memops for AVX-512, so we need to use scalar loads to fill
    // the vector registers for the multiply. This assumes operation on
    // doubles only!

    #ifdef CNN_USE_AVX512
    for (unsigned int i = 0; i < size; i++)
        dst[i] += src[i];
    #else

    for (unsigned int i = 0; i < size/T::unroll_size; i++) {
        typename T::register_type d = T::loadu(&dst[i*T::unroll_size]);
        typename T::register_type s = T::loadu(&src[i*T::unroll_size]);
        T::storeu(&dst[i*T::unroll_size], T::add(d, s));
    }

    for (unsigned int i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        dst[i] += src[i];

    #endif
}

// Set specified number of elements starting from the specified address
// to zeroes.

template<typename T>
inline void setzero_(
    bool                    is_aligned,
    int                     size,
    typename T::value_type* dst)
{

  // Fill a local vector register with all zeroes
  typename T::register_type zero_vec = T::setzero();

  // Vectorize storing zeroes across specified array in memory

  int num_wide_ops  = size / T::unroll_size;
  int remaining_ops = size % T::unroll_size;

  int i;
  for (i = 0; i < num_wide_ops; ++i) {
    typename T::value_type* dest_addr = dst + (i * T::unroll_size);

    if (is_aligned)
      T::store(dest_addr, zero_vec);
    else
      T::storeu(dest_addr, zero_vec);
  }

  // Handle last iteration when size is not evenly divisible by the
  // vector length. In this case, we need to mask off the invalid
  // elements at the end of the vector.

  if (remaining_ops > 0) {
    typename T::mask_type mask_vec = T::generate_mask(remaining_ops);

    typename T::value_type* dest_addr = dst + (i * T::unroll_size);

    T::maskstore(dest_addr, mask_vec, zero_vec);
  }
}

// Vectorization of the inner loop of the computation kernel in
// forward_propagation and backward_propagation. Typically the src vector
// represents the weights and the c value represents the element in the
// input vector. Partial products are stored in the dest vector.

template<typename T>
inline void fmadd_(
    bool                          is_first,
    bool                          is_aligned,
    int                           size,
    const typename T::value_type* src,
    typename T::value_type        c,
    typename T::value_type*       dst)
{

  // Broadcast the scalar value into a vector register to be used as the
  // multiplier constant for all elements in the src vector.
  typename T::register_type c_vec = T::set1(c);

  // Vectorize across output elements, the partial product in the dest
  // vector is added to the product of the weights and the input element.

//  int num_wide_ops  = size / T::unroll_size;
//  int remaining_ops = size % T::unroll_size;

  int num_wide_ops  = (size + T::unroll_size - 1) / T::unroll_size;

  int i;
  for (i = 0; i < num_wide_ops; ++i) {

    // Load weights
    const typename T::value_type* src_addr = src + (i * T::unroll_size);
    typename T::register_type src_vec
        = (is_aligned) ? T::load(src_addr)
        :                T::loadu(src_addr);

    // Load partial products. If this is the first iteration and there
    // are no partial products stored yet, then do not load from the
    // destination array.
    typename T::value_type* dest_addr = dst + (i * T::unroll_size);
    typename T::register_type dest_vec
        = (is_first)   ? T::setzero()
        : (is_aligned) ? T::load(dest_addr)
        :                T::loadu(dest_addr);

    // Multiply input and weights, add to partial product
    dest_vec = T::fmadd(c_vec, src_vec, dest_vec);

    // Store partial products back into results vector
    if (is_aligned)
      T::store(dest_addr, dest_vec);
    else
      T::storeu(dest_addr, dest_vec);

  }

//  // Handle last iteration when size is not evenly divisible by the
//  // vector length. In this case, we need to mask off the invalid
//  // elements at the end of the vector.
//
//  if (remaining_ops > 0) {
//
//    // Generate mask vector
//    typename T::mask_type mask_vec = T::generate_mask(remaining_ops);
//
//    // Load weights
//    const typename T::value_type* src_addr = src + (i * T::unroll_size);
//    typename T::register_type     src_vec  = T::maskload(src_addr, mask_vec);
//
//    // Load partial products
//    typename T::value_type*   dest_addr = dst + (i * T::unroll_size);
//    typename T::register_type dest_vec  = T::maskload(dest_addr, mask_vec);
//
//    // Multiply input and weights, add to partial product
//    dest_vec = T::fmadd(c_vec, src_vec, dest_vec);
//
//    // Store partial products back into results vector
//    T::maskstore(dest_addr, mask_vec, dest_vec);
//
//  }
}

} // namespace detail

#if defined(CNN_USE_AVX512)
#define VECTORIZE_TYPE detail::avx512<T>
#elif defined(CNN_USE_AVX)
#define VECTORIZE_TYPE detail::avx<T>
#elif defined(CNN_USE_SSE)
#define VECTORIZE_TYPE detail::sse<T>
#else
#define VECTORIZE_TYPE detail::generic<T>
#endif

// dst[i] += c * src[i]
template<typename T>
void muladd(const T* src, T c, unsigned int size, T* dst) {
    if (detail::is_aligned(VECTORIZE_TYPE(), src, dst))
        detail::muladd_aligned<VECTORIZE_TYPE>(src, c, size, dst);
    else
        detail::muladd_nonaligned<VECTORIZE_TYPE>(src, c, size, dst);
}

// sum(s1[i] * s2[i])
template<typename T>
T dot(const T* s1, const T* s2, unsigned int size) {
    if (detail::is_aligned(VECTORIZE_TYPE(), s1, s2))
        return detail::dot_product_aligned<VECTORIZE_TYPE>(s1, s2, size);
    else
        return detail::dot_product_nonaligned<VECTORIZE_TYPE>(s1, s2, size);
}

/// dst[i] += src[i]
template<typename T>
void reduce(const T* src, unsigned int size, T* dst) {
    if (detail::is_aligned(VECTORIZE_TYPE(), src, dst))
        return detail::reduce_aligned<VECTORIZE_TYPE>(src, size, dst);
    else
        return detail::reduce_nonaligned<VECTORIZE_TYPE>(src, size, dst);
}

// dst[i] += c * src[i]
template<typename T>
void fmadd(bool is_first, const T* src, T c, unsigned int size, T* dst) {
  bool is_aligned = detail::is_aligned(VECTORIZE_TYPE(), src, dst);
  detail::fmadd_<VECTORIZE_TYPE>(is_first, is_aligned, size, src, c, dst);
}

// dst[i] = 0
template<typename T>
void setzero(unsigned int size, T* dst) {
  bool is_aligned = detail::is_aligned(VECTORIZE_TYPE(), dst);
  detail::setzero_<VECTORIZE_TYPE>(is_aligned, size, dst);
}

} // namespace vectorize
