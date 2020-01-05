/**************************************************************
* NOTE: This code respects the initial interface, so that it
* succesfully passes the online grader. The rest versions
* optimize for speed and space by using floats and 8-bit ints
* 
**************************************************************/ 
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../inc/ising.h"


#define diff 1e-6f
#define WINDOW_SIZE 5
#define MIN_MARGIN 5

// Debugging function that prints the first nXn elements of a sizeXsize array
void print_nn_array(int *x, int n, int size){
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            printf("%d ", x[(i + 20)*size + j + 20]);
        }
        printf("\n");
    }
    printf("\n");
}

// CUDA Kernel
__global__ void computeMoment(int *readArr, int *writeArr, double *weightArr, int n){
    // The dimensions are hardcoded here to simplify extra syntax
    // cuda uses for dynamic shared memory allocation
    __shared__ int readArr_shared[32][32];
    __shared__ double weightArr_shared[5][5];

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

void ising(int *G, double *w, int k, int n)
{
    // Allocate memory for the 3 arrays with cudaMallocManaged()
    // because they will be used inside the kernel
    // The return err values are for debugging only
    int *readArr, *writeArr;
    cudaError_t err1 = cudaMallocManaged(&readArr, n*n*sizeof(int));
    cudaError_t err2 = cudaMallocManaged(&writeArr,n*n*sizeof(int));
    double *weightArr_d;
    cudaError_t er3 = cudaMallocManaged(&weightArr_d, 5*5*sizeof(double));

    // Copy the contents of input arrays inside 
    // the ones we will use inside kernel
    memcpy(readArr, G, n*n*sizeof(int));
    memcpy(weightArr_d, w, 5*5*sizeof(double));

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
        int *temp = readArr;
        readArr = writeArr;
        writeArr = temp;
    }

    //The final result now is in readArr. Copy the contents
    // in array G
    memcpy(G, readArr, n*n*sizeof(int));

    cudaFree( readArr     );
    cudaFree( writeArr 	  );
    cudaFree( weightArr_d );
}

int main()
{
    FILE* fin = fopen("../test/conf-init.bin","rb");
    FILE* fout = fopen("../test/conf-11-3.ans","wb");

    int n=517, k=11;

    double weights[5][5] = { {0.004f,0.016f,0.026f,0.016f,0.004f},
                            {0.016f,0.071f,0.117f,0.071f,0.016f},
                            {0.026f,0.117f,0.000f,0.117f,0.026f},
                            {0.016f,0.071f,0.117f,0.071f,0.016f},
                            {0.004f,0.016f,0.026f,0.016f,0.004f}};


    int *latticeArr = (int *) malloc(n*n*sizeof(int));

    //read from binary
    for (int row=0; row<n; row++)
    {
        for (int col=0; col<n; col++)
        {
            int spin;
            fread(&spin, sizeof(int), 1, fin);
            latticeArr[row*n + col] = spin;
        }
    }

    ising(latticeArr, (double*)weights, k, n);

    //write to binary
    for (int row=0; row<n; row++)
    {
        for (int col=0; col<n; col++)
        {
            int spin = latticeArr[row*n + col];
            fwrite(&spin, sizeof(int), 1, fout);
        }
    }

    free( latticeArr );
    return 0;
}