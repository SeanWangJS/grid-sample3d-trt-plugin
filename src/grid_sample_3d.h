#include <string>

#include <cuda_runtime.h>

#ifndef GRID_SAMPLE_3D_H
#define GRID_SAMPLE_3D_H

enum class GridSample3DInterpolationMode{ Bilinear, Nearest};
enum class GridSample3DPaddingMode{ Zeros, Border, Reflection};
enum class GridSample3DDataType {GFLOAT, GHALF};

template <typename scalar_t>
int grid_sample_3d_cuda(
    const scalar_t* input,
    const scalar_t* grid,
    size_t N, size_t C, size_t D_in, size_t H_in, size_t W_in,
    size_t D_grid, size_t H_grid, size_t W_grid,
    bool align_corners,
    GridSample3DInterpolationMode interpolationMode,
    GridSample3DPaddingMode paddingMode,
    scalar_t* output,
    cudaStream_t stream
);

#endif