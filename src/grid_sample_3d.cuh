// from https://github.com/TrojanXu/onnxparser-trt-plugin-sample/blob/master/TensorRT/plugin/gridSamplerPlugin/gridSampler.cuh

#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "grid_sample_3d.h"

static __forceinline__ __device__
__half operator*(const __half& a, const int& b)
{
    return a * __int2half_rn(b);
}

static __forceinline__ __device__
__half operator*(const float& a, const half& b)
{
    return __float2half(a) * b;
}

static __forceinline__ __device__
__half operator+=(const float& a, const half& b)
{
    return __float2half(a) += b;
}

static __forceinline__ __device__
__half operator/(const __half& a, const int& b)
{
    return a / __int2half_rn(b);
}

static __forceinline__ __device__
__half operator+(const __half& a, const float& b)
{
    return a + __float2half(b);
}

static __forceinline__ __device__
__half operator-(const __half& a, const int& b)
{
    return a - __int2half_rn(b);
}

static __forceinline__ __device__
__half operator-(const int& a, const __half& b)
{
    return __int2half_rn(a) - b;
}

static __forceinline__ __device__
__half operator+=(const __half& a, const __half& b)
{
    return a + b;
}

static __forceinline__ __device__
__half min(const __half& a, const half& b)
{
    return __float2half(min(__half2float(a), __half2float(b)));
}

static __forceinline__ __device__
__half max(const __half& a, const half& b)
{
    return __float2half(max(__half2float(a), __half2float(b)));
}

static __forceinline__ __device__
__half fabs(const __half& a)
{
    //TODO return __habs(a); what happened.
    return __float2half(fabs(__half2float(a)));
}

static __forceinline__ __device__
__half floor(const __half& a)
{
    return hfloor(a);
}

static __forceinline__ __device__
__half roundf(const __half& a)
{
    return hrint(a);
}

static __forceinline__ __device__
__half fmod(const __half& a, const __half& b)
{
  return __float2half(fmodf(__half2float(a), __half2float(b)));
}

// borrow from pytorch aten/src/Aten/native/GridSampler.h
template<typename scalar_t>
static __forceinline__ __device__
scalar_t reflect_coordinates(scalar_t in, int twice_low,
                                           int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = ::fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = ::fmod(in, span);
  int flips = static_cast<int>(::floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

template <typename scalar_t>
static __forceinline__ __device__
scalar_t compute_index(
    const scalar_t coord,
    const int size,
    const GridSample3DPaddingMode padding_mode, 
    const bool align_corners
) {
    // unnormalize coord from [-1, 1] to [0, size - 1] if align_corners = False
    // else unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    scalar_t coord_;
    if(align_corners) {
        coord_ = ((coord + 1.f) / 2) * (size - 1);
    }else {
        coord_ = ((coord + 1.f) * size - 1) / 2;
    }

    // check if the coord_ is out of input boundary,
    // if so, make it back to the boundary based on padding_mode
    if (padding_mode == GridSample3DPaddingMode::Border) { // border mode, clip to [0, size-1]
        coord_ = ::min(static_cast<scalar_t>(size-1), ::max(coord_, static_cast<scalar_t>(0)));
    } else if (padding_mode == GridSample3DPaddingMode::Reflection) { // reflection mode
        if(align_corners) {
            coord_ = reflect_coordinates(coord_, 0, 2 * (size - 1));
        } else {
            coord_ = reflect_coordinates(coord_, -1, 2 * size - 1);
        }
        coord_ = ::min(static_cast<scalar_t>(size-1), ::max(coord_, static_cast<scalar_t>(0)));
    }

    return coord_;
}