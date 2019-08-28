/*
 *	init (IT) inhibition table T
 *	from general (GE) system matrix A of doubles (D)
 *
 */

#ifndef __DGEIT_W_H__
#define __DGEIT_W_H__

void DGEIT_W(double** A, double** Tlocal, int n, int cols, int rank, int cprocs)
{
    int i,j;

    double denAii;

	for (i=0;i<n;i++)
	{
		for (j=0;j<i;j++) // split loop (left part)
		{
			Tlocal[i][j]=0;
		}
		denAii=1/A[i][i]; // calc once for the entire row
		Tlocal[i][i]=denAii;
		for (j=i+1;j<n;j++) // split loop (right part)
		{
			Tlocal[i][j]=0;
		}
		for (j=0;j<n;j++)
		{
			Tlocal[i][j+n]=A[j][i]*denAii;
		}
	}
}

#endif
