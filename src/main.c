#include "ising.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

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
	int8_t *latticeArr = malloc(n*n*sizeof(int8_t));
	
	//read from binary
	for (int row=0; row<n; row++)
		for (int col=0; col<n; col++)
		{
			int spin;
			fread(&spin,sizeof(int),1,fin);
			latticeArr[row*n +col] = (int8_t)spin;
		}
		
	ising(latticeArr,(float*)weights,k,n);
	
	//write to binary
	for (int row=0; row<n; row++)
		for (int col=0; col<n; col++)
		{
			int spin = latticeArr[row*n +col];
			fwrite(&spin,sizeof(int),1,fout);
		}
}
