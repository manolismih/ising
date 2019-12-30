#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define diff 1e-6f

void ising( int8_t *G, float *w, int k, int n)
{
	//convert weight array to float[5][5]
	float (*weightArr)[5] = ( float(*)[5] )w;
	
	//set valid indexes to [-2..2][-2..2]
	weightArr = ( float(*)[5] ) &weightArr[2][2];
	weightArr[0][0] = 0.0f;
	
	int8_t (*readArr) [n] = ( int8_t(*)[n] )G;
	int8_t (*writeArr)[n] = malloc(n*n*sizeof(int8_t));
	
	for (int i=1; i<=k; i++)
	{
		for (int row=0; row<n; row++)
			for (int col=0; col<n; col++)
			{
				float influence = 0.0f;
				for (int i=-2; i<3; i++)
					for (int j=-2; j<3; j++)
					{
						//add extra n so that modulo behaves like mathematics modulo
						//that is return only positive values
						int y = (row+i+n)%n;
						int x = (col+j+n)%n;
						influence += weightArr[i][j]*readArr[y][x];
					}
				writeArr[row][col] = readArr[row][col];
				if (influence < -diff) 	 	writeArr[row][col] = -1;
				else if (influence > diff)	writeArr[row][col] = 1;		
			}
		int8_t (*temp)[n] = readArr;
		readArr = writeArr;
		writeArr = temp;
	}
	if (k%2==1) memcpy(G,readArr,n*n*sizeof(uint8_t));
}
