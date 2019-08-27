/*
 *	zero (ZR) general (GE) matrix mat of rows x cols doubles (D)
 *
 */

#ifndef __DGEZR_H__
#define __DGEZR_H__

void DGEZR(double** mat, int rows, int cols)
{
    int i,j;

	for (i=0;i<rows;i++)
	{
		for (j=0;j<cols;j++)
		{
			mat[i][j]=0;
		}
	}
}

#endif
