#include <stdlib.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <assert.h>
#include <math.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "grid_sample_3d.h"

using half = __half;

void readData(const char* filename, float* data) {
    // todo read data from file line by line and store in data
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "file not found" << std::endl;
        return;
    }

    std::string line;
    int i = 0;
    while (std::getline(file, line)) {
        data[i] = std::stof(line);
        i++;
    }
}

std::string getAbsolutionPath(const std::string relative_path) {
    std::string absolute_path = std::filesystem::absolute(relative_path).string();
    return absolute_path;
}

void testGridSample3dFloat16() {
    size_t N = 1;
    size_t C = 1;
    size_t D_in = 16;
    size_t H_in = 64;
    size_t W_in = 64;
    size_t D_grid = 16;
    size_t H_grid = 64;
    size_t W_grid = 64;

    size_t dim_input[5] = {N, C, D_in, H_in, W_in};
    size_t dim_grid[5] = {N, D_grid, H_grid, W_grid, 3};

    auto input_number = N * C * D_in * H_in * W_in;
    auto input_size = input_number * sizeof(half);
    auto grid_number = N * D_grid * H_grid * W_grid * 3;
    auto grid_size = grid_number * sizeof(half);
    auto output_number = N * C * D_grid * H_grid * W_grid;
    auto output_size = output_number * sizeof(half);

    float* input_fp32 = (float*)malloc(input_number * sizeof(float));
    float* grid_fp32 = (float*)malloc(grid_number * sizeof(float));
    float* output_ref_fp32 = (float*)malloc(output_number * sizeof(float));
    float* output_fp32 = (float*)malloc(output_number * sizeof(float));

    half* input = (half*)malloc(input_size);
    half* grid = (half*)malloc(grid_size);
    half* output = (half*)malloc(output_size);

    const std::string input_path = "../test/data/input.txt";
    const std::string grid_path = "../test/data/grid.txt";
    const std::string output_path = "../test/data/output.txt";
    readData(getAbsolutionPath(input_path).c_str(), input_fp32);
    readData(getAbsolutionPath(grid_path).c_str(), grid_fp32);
    readData(getAbsolutionPath(output_path).c_str(), output_ref_fp32);

    float scale = 1.f;

    for (int i = 0; i < input_number; i++) {
        input[i] = __float2half(input_fp32[i] * scale);
    }

    for (int i = 0; i < grid_number; i++) {
        // grid[i] = __float2half(grid_fp32[i] * 100) / 100.f;
        grid[i] = __float2half(grid_fp32[i]);
    }

    half* d_input;
    half* d_grid;
    half* d_output;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_grid, grid_size);
    cudaMalloc(&d_output, output_size);

    cudaError_t err = cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error in cudaMemcpy: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_grid, grid, grid_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error in cudaMemcpy: %s\n", cudaGetErrorString(err));
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // warmup
    printf("Warmup...\n");
    for(int i = 0; i < 10; i++) {
        grid_sample_3d_cuda<half>(
                            d_input, 
                            d_grid, 
                            N, C, D_in, H_in, W_in,
                            D_grid, H_grid, W_grid,
                            false, 
                            GridSample3DInterpolationMode::Bilinear, 
                            GridSample3DPaddingMode::Zeros,
                            d_output, 
                            stream);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    printf("Run...\n");
    grid_sample_3d_cuda<half>(
                            d_input, 
                            d_grid, 
                            N, C, D_in, H_in, W_in,
                            D_grid, H_grid, W_grid,
                            false, 
                            GridSample3DInterpolationMode::Bilinear, 
                            GridSample3DPaddingMode::Zeros,
                            d_output, 
                            stream);

    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %fms\n", milliseconds);                            

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < output_number; i++) {
        output_fp32[i] = __half2float(output[i]) / scale;
    }

    float max_diff = 0.f;
    int max_index=0;

    for (int i = 0; i < output_number; i++) {
        float diff = output_ref_fp32[i] - output_fp32[i];
        diff = abs(diff);
        if(diff > max_diff) {
            max_diff = diff;
            max_index = i;
        }
    }
    printf("Max error: %f\n", max_diff);
    printf("Max error index: %d\n", max_index);
    printf("Max error value: %f vs %f\n", output_fp32[max_index], output_ref_fp32[max_index]);

    float max_err_rate = max_diff / output_ref_fp32[max_index];
    printf("Max error rate: %f%%\n", max_err_rate * 100);

    cudaFree(d_input);
    cudaFree(d_grid);
    cudaFree(d_output);

    free(input);
    free(grid);
    free(output);
    free(input_fp32);
    free(grid_fp32);
    free(output_fp32);
    free(output_ref_fp32);


    printf("Done\n");
}

void testGridSample3dFloat32() {

    std::cout << "Test GridSample3dFloat32..." << std::endl;

    size_t N = 1;
    size_t C = 1;
    size_t D_in = 16;
    size_t H_in = 64;
    size_t W_in = 64;
    size_t D_grid = D_in;
    size_t H_grid = H_in;
    size_t W_grid = W_in;

    size_t dim_input[5] = {N, C, D_in, H_in, W_in};
    size_t dim_grid[5] = {N, D_grid, H_grid, W_grid, 3};

    auto input_number = N * C * D_in * H_in * W_in;
    auto input_size = input_number * sizeof(float);
    auto grid_number = N * D_grid * H_grid * W_grid * 3;
    auto grid_size = grid_number * sizeof(float);
    auto output_number = N * C * D_grid * H_grid * W_grid;
    auto output_size = output_number * sizeof(float);
    float* input = (float*)malloc(input_size);
    float* grid = (float*)malloc(grid_size);
    float* output_ref = (float*)malloc(output_size);
    float* output = (float*)malloc(output_size);

    const std::string input_path = "../test/data/input.txt";
    const std::string grid_path = "../test/data/grid.txt";
    const std::string output_path = "../test/data/output.txt";
    readData(getAbsolutionPath(input_path).c_str(), input);
    readData(getAbsolutionPath(grid_path).c_str(), grid);
    readData(getAbsolutionPath(output_path).c_str(), output_ref);

    float* d_input;
    float* d_grid;
    float* d_output;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_grid, grid_size);
    cudaMalloc(&d_output, output_size);

    cudaError_t err = cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error in cudaMemcpy: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_grid, grid, grid_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error in cudaMemcpy: %s\n", cudaGetErrorString(err));
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // warmup
    printf("Warmup...\n");
    for(int i = 0; i < 10; i++) {
        grid_sample_3d_cuda<float>(
                            d_input, 
                            d_grid, 
                            N, C, D_in, H_in, W_in,
                            D_grid, H_grid, W_grid,
                            false, 
                            GridSample3DInterpolationMode::Bilinear, 
                            GridSample3DPaddingMode::Zeros,
                            d_output, 
                            stream);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);

    printf("Run...\n");
    grid_sample_3d_cuda<float>(
                            d_input, 
                            d_grid, 
                            N, C, D_in, H_in, W_in,
                            D_grid, H_grid, W_grid,
                            false, 
                            GridSample3DInterpolationMode::Bilinear, 
                            GridSample3DPaddingMode::Zeros,
                            d_output, 
                            stream);

    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %fms\n", milliseconds);                            

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    float max_diff = 0.f;
    int max_index=0;
    float min = 0.f;
    for (int i = 0; i < output_number; i++) {
        float diff = output_ref[i] - output[i];
        diff = abs(diff);
        if (diff > max_diff) {
            max_diff = diff;
            max_index = i;
        }
    }


    printf("Max error: %f\n", max_diff);
    printf("Max error index: %d\n", max_index);
    printf("Max error value: %f vs %f\n", output[max_index], output_ref[max_index]);
    printf("Max error rate: %f%%\n", max_diff / output_ref[0] * 100);

    cudaFree(d_input);
    cudaFree(d_grid);
    cudaFree(d_output);

    free(input);
    free(grid);
    free(output_ref);
    free(output);
    printf("Done\n");
}


int main(int argc, char** argv) {
    // testGridSample3dFloat16();
    testGridSample3dFloat32();
    
    return 0;

}

