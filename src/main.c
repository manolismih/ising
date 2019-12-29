#include "ising.h"
#include <stdio.h>
#include <stdlib.h>

int main()
{
	FILE* fin = fopen("test/conf-init.bin","rb");
	FILE* fout = fopen("test/conf-11.ans","wb");
	int n=517, k=11;
	double weights[5][5] = { {0.004,0.016,0.026,0.016,0.004},
							 {0.016,0.071,0.117,0.071,0.016},
							 {0.026,0.117,0.000,0.117,0.026},
							 {0.016,0.071,0.117,0.071,0.016},
							 {0.004,0.016,0.026,0.016,0.004}};
	int *latticeArr = malloc(n*n*sizeof(int));
	fread(latticeArr,sizeof(int),n*n,fin);
	ising(latticeArr,(double*)weights,k,n);
	fwrite(latticeArr,sizeof(int),n*n,fout);
}
