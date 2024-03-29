#ifndef __GEZR_H__
#define __GEZR_H__

/*
 *	zero (ZR) general (GE) matrix mat of rows x cols
 *
 */

void dgezr (	double** mat,
				int rows,
				int cols	)
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

void sgezr (	float** mat,
				int rows,
				int cols	)
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
