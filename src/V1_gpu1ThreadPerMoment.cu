/**************************************************************
 * NOTE: This code respects the initial interface, so that it
 * succesfully passes the online grader. The rest versions
 * optimize for speed and space by using floats and 8-bit ints
 * 
 **************************************************************/ 
#include <stdlib.h>
#include <string.h>

#include <stdio.h>

#define diff 1e-6f

__global__ void computeMoment(int *readArr, int *writeArr, double weightArr[5][5], int n){
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;

	float influence = 0.0f;
	for (int i=-2; i<3; i++)
	{
		for (int j=-2; j<3; j++)
		{
			//add extra n so that modulo behaves like mathematics modulo
			//that is return only positive values
			int y = (row+i+n)%n;
			int x = (col+j+n)%n;
			influence += weightArr[i][j]*readArr[y*n + x];
		}
	}

	writeArr[row*n + col] = readArr[row*n + col];
	if 		(influence<-diff)	writeArr[row*n + col] = -1;
	else if (influence>diff) 	writeArr[row*n + col] = 1;
	__syncthreads();

}

void ising(int *G, double *w, int k, int n)
{
	printf("In ising\n");
	//convert weight array to double[5][5]
	double (*weightArr)[5] = ( double(*)[5] )w;
	
	//set valid indexes to [-2..2][-2..2]
	weightArr = ( double(*)[5] ) &weightArr[2][2];
	weightArr[0][0] = 0.0;
	
	// int (*readArr) [n] = ( int(*)[n] )G;
	// int (*writeArr)[n] = ( int(*)[n] ) malloc(n*n*sizeof(int));

	// int *readArr = G;
	// int *writeArr = (int *) malloc(n*n*sizeof(int));

	int *readArr, *writeArr;
	cudaError_t err = cudaMallocManaged(&readArr, n*n*sizeof(int));
	printf("%d\n", err);
	cudaError_t er = cudaMallocManaged(&writeArr,n*n*sizeof(int));
	printf("%d\n", er);

	memcpy(readArr, G, n*n*sizeof(int));
	
	for (int i=1; i<=k; i++)
	{
		dim3 dimBlock(n, n);
		computeMoment<<<1, dimBlock>>> (readArr, writeArr, weightArr, n);

		// Wait for GPU to finish before accessing on host
		cudaDeviceSynchronize();

		int *temp = readArr;
		readArr = writeArr;
		writeArr = temp;
	}
	printf("Leaving ising\n");

	// memcpy(G, readArr, n*n*sizeof(int));
	G = readArr;

	cudaFree( readArr  );
	cudaFree( writeArr );
}

#include "../inc/ising.h"
#include <stdlib.h>
#include <stdint.h>

int main()
{
	FILE* fin = fopen("../test/conf-init.bin","rb");
	FILE* fout = fopen("../test/conf-11.ans","wb");


	int n=6, k=5;

	float weights[5][5] = {  {0.004f,0.016f,0.026f,0.016f,0.004f},
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
			// int spin;
			// fread(&spin, sizeof(int), 1, fin);
			latticeArr[row*n +col] = -1;
			printf("%d ", latticeArr[row*n + col]);

		}
		printf("\n");
	}
	printf("\n\n");


	ising(latticeArr, (double*)weights, k, n);
	
	//write to binary
	for (int row=0; row<n; row++)
	{
		for (int col=0; col<n; col++)
		{
			int spin = latticeArr[row*n + col];
			printf("%d ", spin);
			fwrite(&spin, sizeof(int), 1, fout);
		}
		printf("\n");
	}

	printf("Exiting!\n");
	return 0;
}
