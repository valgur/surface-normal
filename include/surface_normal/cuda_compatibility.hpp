#pragma once

#include <cmath>

#ifdef __CUDACC__
#  include <cuda.h>
#else
#  define __host__
#  define __device__
#  define __forceinline__ inline
using std::sqrt;
inline float rsqrt(float x) { return 1.f / sqrt(x); }
#endif

#ifdef WITH_CUDA
#  define USE_CUDA true
#else
#  define USE_CUDA false
#endif

#define __hdi__ __host__ __device__ __forceinline__
