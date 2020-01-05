 #include <stdlib.h>
 #include <string.h>
 #include "ising.h"

 // CUDA Kernel
 __global__ void computeMoment(int8_t *readArr, int8_t *writeArr, float *weightArr, int n, int tileSize){
     int row_init = blockIdx.x*(blockDim.x*tileSize) + threadIdx.x*tileSize;
     int col_init = blockIdx.y*(blockDim.y*tileSize) + threadIdx.y*tileSize;
 
     // Assign each thread a tileSizeXtileSize tile
     for(int ii=0; ii<tileSize; ++ii){
         for (int jj=0; jj<tileSize; ++jj){
             int row = row_init + ii;
             int col = col_init + jj;
 
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
                         int y = (row+i+n)%n;
                         int x = (col+j+n)%n;
                         influence += weightArr[i*5 + j]*readArr[y*n + x];
                     }
                 }
             
                 writeArr[row*n + col] = readArr[row*n + col];
                 if 	(influence<-diff)	writeArr[row*n + col] = -1;
                 else if (influence>diff)	writeArr[row*n + col] = 1;
                 __syncthreads();
             }
         }
     }
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
 
 
     //set valid indexes to [-2..2][-2..2]
     weightArr_d = &weightArr_d[2*5 + 2];
     weightArr_d[0] = 0.0;
 
     // Define the thread tile size, that is the size of the block of
     // moments a single thread will calculate. Set it to 5x5
     int tileSize = 5;
 
     for (int i=1; i<=k; i++)
     {
         // Create blocks of size 32x32 threads per block
         // The number of blocks will adjust to fit the input n
         dim3 dimBlock(32, 32);
         int gridSz = (n + 32*tileSize)/ 32*tileSize;
         dim3 dimGrid(gridSz, gridSz);
 
         // Run the kernel in GPU
         computeMoment<<<dimGrid, dimBlock>>> (readArr, writeArr, weightArr_d, n, tileSize);
 
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
     memcpy(G, readArr, n*n*sizeof(int));
 
     cudaFree( readArr     );
     cudaFree( writeArr 	  );
     cudaFree( weightArr_d );
 }
