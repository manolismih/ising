#include <stdlib.h>
#include <string.h>
#include "ising.h"

#define WINDOW_SIZE 5
#define MIN_MARGIN 5

// CUDA Kernel
__global__ void computeMoment(int8_t *readArr, int8_t *writeArr, float *weightArr, int n){
    // The dimensions are hardcoded here to simplify extra syntax
    // cuda uses for dynamic shared memory allocation
    __shared__ int8_t readArr_shared[32][32];
    __shared__ float weightArr_shared[5][5];

    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    if(threadIdx.x<5 && threadIdx.y < 5){
        weightArr_shared[threadIdx.x][threadIdx.y] = weightArr[threadIdx.x*WINDOW_SIZE + threadIdx.y];
    }
    __syncthreads();

    // Only values within the below borders will be used but the __syncthreads()
    // function has to be called outside if statements so we load everything here
    readArr_shared[threadIdx.x][threadIdx.y] = readArr[row*n + col];
    __syncthreads();

    // If coordinates are between boundaries
    // update the write array accordingly
    if(row < n && col < n){
        float influence = 0.0f;
        for (int i=-2; i<3; i++)
        {
            for (int j=-2; j<3; j++)
            {
                //add extra n so that modulo behaves like mathematics modulo
                //that is return only positive values
                if(threadIdx.x >= MIN_MARGIN && threadIdx.y >= MIN_MARGIN && 
                    threadIdx.x <= 31-MIN_MARGIN && threadIdx.y <= 31-MIN_MARGIN){
                    int y = threadIdx.x + i;
                    int x = threadIdx.y + j;
                    influence += weightArr_shared[i+2][j+2]*readArr_shared[y][x];
                }else{
                    int y = (row+i+n)%n;
                    int x = (col+j+n)%n;
                    influence += weightArr_shared[i+2][j+2]*readArr[y*n + x];
                }
            }
        }

        if(threadIdx.x >= MIN_MARGIN && threadIdx.y >= MIN_MARGIN && 
            threadIdx.x <= 31-MIN_MARGIN && threadIdx.y <= 31-MIN_MARGIN){
            writeArr[row*n + col] = readArr_shared[threadIdx.x][threadIdx.y];
            if 		(influence<-diff)	writeArr[row*n + col] = -1;
            else if (influence>diff)	writeArr[row*n + col] = 1;
        }else {
            writeArr[row*n + col] = readArr[row*n + col];
            if 		(influence<-diff)	writeArr[row*n + col] = -1;
            else if (influence>diff)	writeArr[row*n + col] = 1;
        }
    }
    __syncthreads();

}

void ising(int8_t *G, float *w, int k, int n)
{
    // Allocate memory for the 3 arrays with cudaMallocManaged()
    // because they will be used inside the kernel
    // The return err values are for debugging only
    int8_t *readArr, *writeArr;
    cudaError_t err1 = cudaMallocManaged(&readArr, n*n*sizeof(int8_t));
    cudaError_t err2 = cudaMallocManaged(&writeArr,n*n*sizeof(int8_t));
    float *weightArr_d;
    cudaError_t er3 = cudaMallocManaged(&weightArr_d, 5*5*sizeof(float));

    // Copy the contents of input arrays inside 
    // the ones we will use inside kernel
    memcpy(readArr, G, n*n*sizeof(int8_t));
    memcpy(weightArr_d, w, 5*5*sizeof(float));

    for (int i=1; i<=k; i++)
    {
        // Create blocks of size 32x32 threads per block
        // The number of blocks will adjust to fit the input n
        dim3 dimBlock(32, 32);
        int gridSz = (n + 32)/ 32;
        dim3 dimGrid(gridSz, gridSz);

        // Run the kernel in GPU
        computeMoment<<<dimGrid, dimBlock>>> (readArr, writeArr, weightArr_d, n);

        // Uncomment below to check for launch errors
        //printf("%s\n", cudaGetErrorString(cudaGetLastError()));

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        // Swap read and write arrays
        int8_t *temp = readArr;
        readArr = writeArr;
        writeArr = temp;
    }

    //The final result now is in readArr. Copy the contents
    // in array G
    memcpy(G, readArr, n*n*sizeof(int8_t));

    cudaFree( readArr     );
    cudaFree( writeArr 	  );
    cudaFree( weightArr_d );
}
