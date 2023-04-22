
#include <gputk.h>
#include <stdlib.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
#define TILE_SIZE 16
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int threadY = threadIdx.y;
  int threadX = threadIdx.x;
  int blockX = blockIdx.x;
  int blockY = blockIdx.y;

  int Row = blockY*blockDim.y+threadY;
  int Col = blockX*blockDim.x+threadX;

  __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
  __shared__ float shared_b[TILE_SIZE][TILE_SIZE];

  __shared__ float shared_a_2[TILE_SIZE][TILE_SIZE];
  __shared__ float shared_b_2[TILE_SIZE][TILE_SIZE];

  float sum = 0;

  int numTiles = (int)ceil((numAColumns * 1.0) / TILE_SIZE);

  bool lessARows = Row < numARows;
  bool lessBColumns = Col < numBColumns;
  bool large = numCRows >= 2000 && numCColumns >= 2000;

  for(int i = 0; i < numTiles; i+=1) {
     int tileBytes = i*TILE_SIZE;
     int rowBytes = Row*numAColumns;

     if(lessARows && tileBytes + threadX < numAColumns) {
         shared_a[threadY][threadX] = A[rowBytes + tileBytes + threadX];
     } else {
         shared_a[threadY][threadX] = 0.0;
     }

     if(lessBColumns && tileBytes + threadY < numBRows) {
         shared_b[threadY][threadX] = B[(tileBytes + threadY) * numBColumns + Col];
     } else {
         shared_b[threadY][threadX] = 0.0;
     }

     if(large && lessARows && tileBytes + threadX + TILE_SIZE < numAColumns) {
        shared_a_2[threadY][threadX] = A[rowBytes + tileBytes + threadX + TILE_SIZE];
     } else {
        shared_a_2[threadY][threadX] = 0.0;
     }

     if(large && lessBColumns && tileBytes + threadY + TILE_SIZE < numBRows) {
        shared_b_2[threadY][threadX] = B[(tileBytes + threadY + TILE_SIZE) * numBColumns + Col];
     } else {
        shared_b_2[threadY][threadX] = 0.0;
     }

      __syncthreads();

      for(int k = 0; k < TILE_SIZE; k+=1) {
          sum = sum + shared_a[threadY][k] * shared_b[k][threadX] + shared_a_2[threadY][k] * shared_b_2[k][threadX];
          k += 1;
          sum = sum + shared_a[threadY][k] * shared_b[k][threadX] + shared_a_2[threadY][k] * shared_b_2[k][threadX];
          k += 1;
          sum = sum + shared_a[threadY][k] * shared_b[k][threadX] + shared_a_2[threadY][k] * shared_b_2[k][threadX];
          k += 1;
          sum = sum + shared_a[threadY][k] * shared_b[k][threadX] + shared_a_2[threadY][k] * shared_b_2[k][threadX];
          k+= 1;
          sum = sum + shared_a[threadY][k] * shared_b[k][threadX] + shared_a_2[threadY][k] * shared_b_2[k][threadX];
          k += 1;
          sum = sum + shared_a[threadY][k] * shared_b[k][threadX] + shared_a_2[threadY][k] * shared_b_2[k][threadX];
          k += 1;
          sum = sum + shared_a[threadY][k] * shared_b[k][threadX] + shared_a_2[threadY][k] * shared_b_2[k][threadX];
          k += 1;
          sum = sum + shared_a[threadY][k] * shared_b[k][threadX] + shared_a_2[threadY][k] * shared_b_2[k][threadX];
      }

       __syncthreads();

       if(large) {
        i+=1;
       }
  }

  if(lessARows && lessBColumns)
    C[Row*numCColumns+Col] = sum;
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");
  hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceA, numARows*numAColumns*sizeof(float));
  cudaMalloc((void**)&deviceB, numBRows*numBColumns*sizeof(float));
  cudaMalloc((void**)&deviceC, numCRows*numCColumns*sizeof(float));

  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  //transpose(hostB, numBRows, numBColumns);
  cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(float), cudaMemcpyHostToDevice);
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 blocks((int)ceil(numCColumns / (TILE_SIZE*1.0)), (int)ceil(numCRows/(TILE_SIZE*1.0)));
  dim3 threads(TILE_SIZE, TILE_SIZE);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<blocks, threads>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost);
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");


  // TRANSPOSE C BACK
  // transpose(C, numCRows, numCColumns);

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory herei
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}