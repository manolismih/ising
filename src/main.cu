#include "ising.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

int main()
{
	FILE* fin = fopen("test/conf-init.bin","rb");
	FILE* fout = fopen("test/conf-11.ans","wb");
	int n=517, k=11;
	float weights[5][5] = {  {0.004f,0.016f,0.026f,0.016f,0.004f},
							 {0.016f,0.071f,0.117f,0.071f,0.016f},
							 {0.026f,0.117f,0.000f,0.117f,0.026f},
							 {0.016f,0.071f,0.117f,0.071f,0.016f},
							 {0.004f,0.016f,0.026f,0.016f,0.004f}};
	int8_t *latticeArr = ( int8_t* ) malloc(n*n*sizeof(int8_t));
	
	//read from binary
	for (int row=0; row<n; row++)
		for (int col=0; col<n; col++)
		{
			int spin;
			fread(&spin,sizeof(int),1,fin);
			latticeArr[row*n +col] = (int8_t)spin;
		}
		
	double elapsedTime=0.0;
	struct timeval start, end;

	gettimeofday(&start, NULL);
	ising(latticeArr,(float*)weights,k,n); // !!!Actual computation
	gettimeofday(&end, NULL);
	
	elapsedTime += end.tv_sec -start.tv_sec +(end.tv_usec-start.tv_usec)/1e6;
	printf("\nComputations done in %lf seconds\n", elapsedTime);
	
	//write to binary
	for (int row=0; row<n; row++)
		for (int col=0; col<n; col++)
		{
			int spin = latticeArr[row*n +col];
			fwrite(&spin,sizeof(int),1,fout);
		}
	return 0;
}
