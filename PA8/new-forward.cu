#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 20
#define TILE_WIDTH2 17
#define FILTER_WIDTH 7

__constant__ float filters[7*7*16*4];

// each thread calculates 4 outputs
__global__ void conv_forward_kernel(float *y, const float* __restrict__ x, const float* k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) filters[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    __shared__ float shmem[86][86];

     int tx = 4*threadIdx.x, ty = 4*threadIdx.y, b = blockIdx.x, m = blockIdx.y;

     int h = ty, w = tx;

     for(int i = threadIdx.y; i < 86; i += 20) {
        for(int j = threadIdx.x; j < 86; j += 20) {
            shmem[i][j] = x4d(b, 0, i, j);
        }
     }

     __syncthreads();

      // load into registers
     float input[10][10];

     #pragma unroll
     for(int i = 0; i < 10; i++) {
        #pragma unroll
        for(int j = 0; j < 10; j++) {
            int r = h+i, c = w+j;
            input[i][j] = shmem[r][c];
        }
     }

     float results[4][4];

     for(int i = 0; i < 4; i++) {
         for(int j = 0; j < 4; j++) {
             results[i][j] = 0.0f;
         }
     }

    #pragma unroll
     for(int r = 0; r < FILTER_WIDTH; r++) {
        #pragma unroll
        for(int col = 0; col < FILTER_WIDTH; col++) {
            float zero = k4d(m, 0, r, col);

            #pragma unroll
            for(int i = 0; i < 4; i++) {
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    results[i][j] += input[i+r][j+col] * zero;
                }
            }
         }
    }

    #pragma unroll
    for(int i = 0; i < 4; i++) {
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            y4d(b, m, h+i, w+j) = results[i][j];
        }
    }

#undef y4d
#undef x4d
#undef k4d
}

// each thread here calculated 4 output values
__global__ void conv_forward_kernel2(float *y, const float* __restrict__ x, const float* k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) filters[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    __shared__ float shmem[4][40][40];

     int b = blockIdx.x, m = blockIdx.y, tx = 2*threadIdx.x, ty = 2*threadIdx.y;

     int h = ty, w = tx;

     for(int i = threadIdx.y; i < 40; i += TILE_WIDTH2) {
        #pragma unroll
        for(int j = threadIdx.x; j < 40; j += TILE_WIDTH2) {
            shmem[0][i][j] = x4d(b, 0, i, j);
            shmem[1][i][j] = x4d(b, 1, i, j);
            shmem[2][i][j] = x4d(b, 2, i, j);
            shmem[3][i][j] = x4d(b, 3, i, j);
        }
     }

     __syncthreads();

     // load into registers
     float input[4][8][8];

     #pragma unroll
     for(int i = 0; i < 8; i++) {
        #pragma unroll
        for(int j = 0; j < 8; j++) {
            int r = h+i, c = w+j;
            input[0][i][j] = shmem[0][r][c];
            input[1][i][j] = shmem[1][r][c];
            input[2][i][j] = shmem[2][r][c];
            input[3][i][j] = shmem[3][r][c];
        }
     }

     float accUpperLeft = 0.0f, accBottomRight = 0.0f, accBottomLeft = 0.0f, accTopRight = 0.0f;

    #pragma unroll
     for(int r = 0; r < FILTER_WIDTH; r++) {
        #pragma unroll
        for(int col = 0; col < FILTER_WIDTH; col++) {
            float zero = k4d(m, 0, r, col), one = k4d(m, 1, r, col), two = k4d(m, 2, r, col), three = k4d(m, 3, r, col);

            int nextR = r+1, nextC = col+1;

            accBottomLeft += input[0][nextR][col] * zero;
            accBottomLeft += input[1][nextR][col] * one;
            accBottomLeft += input[2][nextR][col] * two;
            accBottomLeft += input[3][nextR][col] * three;

            accTopRight += input[0][r][nextC] * zero;
            accTopRight += input[1][r][nextC] * one;
            accTopRight += input[2][r][nextC] * two;
            accTopRight += input[3][r][nextC] * three;

            accUpperLeft += input[0][r][col] * zero;
            accUpperLeft += input[1][r][col] * one;
            accUpperLeft += input[2][r][col] * two;
            accUpperLeft += input[3][r][col] * three;

            accBottomRight += input[0][nextR][nextC] * zero;
            accBottomRight += input[1][nextR][nextC] * one;
            accBottomRight += input[2][nextR][nextC] * two;
            accBottomRight += input[3][nextR][nextC] * three;
        }
    }

    y4d(b, m, h, w) = accUpperLeft;
    y4d(b, m, h+1, w+1) = accBottomRight;
    y4d(b, m, h+1, w) = accBottomLeft;
    y4d(b, m, h, w+1) = accTopRight;

#undef y4d
#undef x4d
#undef k4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc(device_x_ptr, B*C*H*W*sizeof(float));
    cudaMalloc(device_y_ptr, B*M*(W-K+1)*(H-K+1)*sizeof(float));
    cudaMalloc(device_k_ptr, M*C*K*K*sizeof(float));

    cudaMemcpy(*device_x_ptr, host_x, B*C*H*W*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, host_k, M*C*K*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(filters, host_k, K*K*M*C*sizeof(float));
}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    if(M < 16) {
        dim3 blockDim(20, 20, 1);

        dim3 gridDim(B,M,1);

        conv_forward_kernel<<<gridDim, blockDim>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    } else {
        dim3 blockDim(TILE_WIDTH2, TILE_WIDTH2, 1);

        dim3 gridDim(B,M,1);

        conv_forward_kernel2<<<gridDim, blockDim>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_y, device_y, B*M*(H-K+1)*(W-K+1)*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}