#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE
__constant__ float mask[Mask_width * Mask_width];

__global__ void convolution(float * input, const float * __restrict__ M, float * output, int channels, int width, int height) {
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    int col_input = col - Mask_radius, row_input = row - Mask_radius;

    __shared__ float input_S[w][w][3];

    if(col_input >= 0 && row_input >= 0 && col_input < width && row_input < height) {
        input_S[threadIdx.y][threadIdx.x][0] = input[(row_input*width+col_input)*channels];
        input_S[threadIdx.y][threadIdx.x][1] = input[(row_input*width+col_input)*channels+1];
        input_S[threadIdx.y][threadIdx.x][2] = input[(row_input*width+col_input)*channels+2];
    } else {
        input_S[threadIdx.y][threadIdx.x][0] = 0.0f;
        input_S[threadIdx.y][threadIdx.x][1] = 0.0f;
        input_S[threadIdx.y][threadIdx.x][2] = 0.0f;
    }

    __syncthreads();

    if(threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH) {
        for(int channel = 0; channel < channels; channel++) {
            float accum = 0.0f;

            for(int i = 0; i < Mask_width; i++) { // row
                for(int j = 0; j < Mask_width; j++) { // col
                    accum = accum + mask[i*Mask_width+j]*input_S[i+threadIdx.y][j+threadIdx.x][channel];
                }
            }


            if(row < height && col < width)
                output[(row*width+col)*channels + channel] = clamp(accum);
        }
    }

    __syncthreads();
}

int main(int argc, char *argv[]) {
  gpuTKArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  gpuTKImage_t inputImage;
  gpuTKImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = gpuTKArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = gpuTKArg_getInputFile(arg, 0);
  inputMaskFile  = gpuTKArg_getInputFile(arg, 1);

  inputImage   = gpuTKImport(inputImageFile);
  hostMaskData = (float *)gpuTKImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth    = gpuTKImage_getWidth(inputImage);
  imageHeight   = gpuTKImage_getHeight(inputImage);
  imageChannels = gpuTKImage_getChannels(inputImage);

  outputImage = gpuTKImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = gpuTKImage_getData(inputImage);
  hostOutputImageData = gpuTKImage_getData(outputImage);

  gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  cudaMalloc(&deviceInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc(&deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc(&deviceMaskData, maskRows*maskColumns*sizeof(float));

  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  gpuTKTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, maskRows*maskColumns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, hostMaskData, Mask_width*Mask_width*sizeof(float));

  gpuTKTime_stop(Copy, "Copying data to the GPU");

  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE

  dim3 dimGrid(ceil((imageWidth*1.0)/TILE_WIDTH), ceil((imageHeight*1.0)/TILE_WIDTH)), dimBlock(w, w);
  convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,
                                     deviceOutputImageData, imageChannels,
                                     imageWidth, imageHeight);

  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  gpuTKTime_start(Copy, "Copying data from the GPU");

  //@@ INSERT CODE HERE
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost);

  gpuTKTime_stop(Copy, "Copying data from the GPU");

  gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKSolution(arg, outputImage);

  //@@ Insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  //free(hostInputImageData);
  //free(hostOutputImageData);
  free(hostMaskData);

  gpuTKImage_delete(outputImage);
  gpuTKImage_delete(inputImage);

  return 0;
}