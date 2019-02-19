/******************************************************************************
** Copyright (c) 2019, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/

#include "libxsmm_rng.h"
#include <libxsmm_intrinsics_x86.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/** 2048-bit state for RNG */
LIBXSMM_APIVAR(unsigned int internal_rng_state0[16]);
LIBXSMM_APIVAR(unsigned int internal_rng_state1[16]);
LIBXSMM_APIVAR(unsigned int internal_rng_state2[16]);
LIBXSMM_APIVAR(unsigned int internal_rng_state3[16]);


#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH) /* __AVX512F__ */
LIBXSMM_API_INLINE void libxsmm_rng_state_load_avx512(void)
{
  libxsmm_intrinsics_mm512_rng_state0 = _mm512_loadu_si512(internal_rng_state0);
  libxsmm_intrinsics_mm512_rng_state1 = _mm512_loadu_si512(internal_rng_state1);
  libxsmm_intrinsics_mm512_rng_state2 = _mm512_loadu_si512(internal_rng_state2);
  libxsmm_intrinsics_mm512_rng_state3 = _mm512_loadu_si512(internal_rng_state3);
}

LIBXSMM_API_INLINE void libxsmm_rng_state_store_avx512(void)
{
  _mm512_storeu_si512(internal_rng_state0, libxsmm_intrinsics_mm512_rng_state0);
  _mm512_storeu_si512(internal_rng_state1, libxsmm_intrinsics_mm512_rng_state1);
  _mm512_storeu_si512(internal_rng_state2, libxsmm_intrinsics_mm512_rng_state2);
  _mm512_storeu_si512(internal_rng_state3, libxsmm_intrinsics_mm512_rng_state3);
}
#endif


LIBXSMM_API_INLINE void libxsmm_rng_float_jump(uint32_t* state0, uint32_t* state1, uint32_t* state2, uint32_t* state3)
{
  static const uint32_t jump_table[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };
  uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int i, b;

  LIBXSMM_ASSERT(4 == sizeof(jump_table) / sizeof(*jump_table));
  for (i = 0; i < 4; ++i) {
    for (b = 0; b < 32; ++b) {
      if (jump_table[i] & (UINT32_C(1) << b)) {
        s0 ^= *state0;
        s1 ^= *state1;
        s2 ^= *state2;
        s3 ^= *state3;
      }
      { /* draw one more integer */
        const uint32_t t = *state1 << 9;
        *state2 ^= *state0;
        *state3 ^= *state1;
        *state1 ^= *state2;
        *state0 ^= *state3;
        *state2 ^= t;
        *state3 = ((*state3 << 11) | (*state3 >> (32 - 11)));
      }
    }
  }
  *state0 = s0;
  *state1 = s1;
  *state2 = s2;
  *state3 = s3;
}


LIBXSMM_API_INLINE float libxsmm_rng_scalar_float_next(int i)
{
  const uint32_t rng_mantissa = (internal_rng_state0[i] + internal_rng_state3[i]) >> 9;
  const uint32_t t = internal_rng_state1[i] << 9;
  union { uint32_t i; float f; } rng;

  internal_rng_state2[i] ^= internal_rng_state0[i];
  internal_rng_state3[i] ^= internal_rng_state1[i];
  internal_rng_state1[i] ^= internal_rng_state2[i];
  internal_rng_state0[i] ^= internal_rng_state3[i];
  internal_rng_state2[i] ^= t;
  internal_rng_state3[i] = ((internal_rng_state3[i] << 11) | (internal_rng_state3[i] >> (32 - 11)));

  rng.i = 0x3f800000 | rng_mantissa;
  return rng.f - 1.0f;
}


LIBXSMM_API void libxsmm_rng_set_seed(unsigned int/*uint32_t*/ seed)
{
  static const uint32_t temp_state[] = {
     31,  30,  29,  28,  27,  26,  25,  24,  23,  22,  21,  20,  19,  18,  17,  16,
    131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116,
    231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216,
    331, 330, 329, 328, 327, 326, 325, 324, 323, 322, 321, 320, 319, 318, 317, 316
  };
  libxsmm_blasint i;

  /* finish initializing the state */
  LIBXSMM_ASSERT((16 * 4) == sizeof(temp_state) / sizeof(*temp_state));
  for (i = 0; i < 16; ++i) {
    internal_rng_state0[i] = seed + temp_state[i];
    internal_rng_state1[i] = seed + temp_state[i+16];
    internal_rng_state2[i] = seed + temp_state[i+32];
    internal_rng_state3[i] = seed + temp_state[i+48];
  }
  for (i = 0; i < 16; ++i) {
    libxsmm_rng_float_jump(/* progress each sequence by 2^64 */
      internal_rng_state0 + i, internal_rng_state1 + i,
      internal_rng_state2 + i, internal_rng_state3 + i);
  }
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH) /* __AVX512F__ */
  libxsmm_rng_state_load_avx512(); /* bring scalar state to AVX-512 */
#endif
  /* for consistency, other RNGs are seeded as well */
#if !defined(_WIN32) && !defined(__CYGWIN__) && (defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
  srand48(seed);
#endif
  srand(seed);
}


LIBXSMM_API void libxsmm_rng_f32_seq(float* rngs, libxsmm_blasint count)
{
  libxsmm_blasint i = 0;
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH) /* __AVX512F__ */
  const libxsmm_blasint n = (count / 16) * 16; /* multiple of vector-length */
  for (; i < n; i += 16) {
    _mm512_storeu_ps(rngs+i, LIBXSMM_INTRINSICS_MM512_RNG_PS());
  }
  if (i < count) { /* remainder */
    libxsmm_rng_state_store_avx512(); /* bring AVX-512 state to scalar */
#else
  {
#endif
    for (; i < count; ++i) { /* scalar */
      rngs[i] = libxsmm_rng_scalar_float_next(LIBXSMM_MOD2(i, 16));
    }
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH) /* __AVX512F__ */
    libxsmm_rng_state_load_avx512(); /* bring scalar state to AVX-512 */
#endif
  }
}


LIBXSMM_API unsigned int libxsmm_rng_u32(unsigned int n)
{
#if defined(_WIN32) || defined(__CYGWIN__) || !(defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
  const unsigned int rand_max1 = (unsigned int)(RAND_MAX)+1U;
  const unsigned int q = (rand_max1 / n) * n;
  unsigned int r = (unsigned int)rand();
  if (q != rand_max1)
#else
  const unsigned int q = ((1U << 31) / n) * n;
  unsigned int r = (unsigned int)lrand48();
  if (q != (1U << 31))
#endif
  {
#if defined(_WIN32) || defined(__CYGWIN__) || !(defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
    while (q <= r) r = (unsigned int)rand();
#else
    while (q <= r) r = (unsigned int)lrand48();
#endif
  }
  return r % n;
}


LIBXSMM_API double libxsmm_rng_f64(void)
{
#if defined(_WIN32) || defined(__CYGWIN__) || !(defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
  static const double scale = 1.0 / (RAND_MAX);
  return scale * (double)rand();
#else
  return drand48();
#endif
}
