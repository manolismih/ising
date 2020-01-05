/**************************************************************
 * NOTE: This code respects the initial interface, so that it
 * succesfully passes the online grader. The rest versions
 * optimize for speed and space by using floats and 8-bit ints
 * 
 **************************************************************/ 
#include <stdlib.h>
#include <string.h>

#define diff 1e-6f

void ising( int *G, double *w, int k, int n)
{
	//convert weight array to double[5][5]
	double (*weightArr)[5] = ( double(*)[5] )w;
	
	//set valid indexes to [-2..2][-2..2]
	weightArr = ( double(*)[5] ) &weightArr[2][2];
	weightArr[0][0] = 0.0;
	
	int (*readArr) [n] = ( int(*)[n] )G;
	int (*writeArr)[n] = malloc(n*n*sizeof(int));
	
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
				if 		(influence<-diff)	writeArr[row][col] = -1;
				else if (influence>diff) 	writeArr[row][col] = 1;		
			}
		int (*temp)[n] = readArr;
		readArr = writeArr;
		writeArr = temp;
	}
	if (k%2==1) memcpy(G,readArr,n*n*sizeof(int));
}
