/*
 *	init (IT) inhibition table T
 *	from general (GE) system matrix A of doubles (D)
 *
 */

void DGEIT_W_1D(double* A, double* T, int n)
{
    int i,j;
    int d=2*n;

    double denAii;

	for (i=0;i<n;i++)
	{
		for (j=0;j<i;j++) // split loop (left part)
		{
			T[i*d+j]=0;
		}
		denAii=1/A[i*n+i]; // calc once for the entire row
		T[i*d+i]=denAii;
		for (j=i+1;j<n;j++) // split loop (right part)
		{
			T[i*d+j]=0;
		}
		for (j=0;j<n;j++)
		{
			T[i*d+j+n]=A[j*n+i]*denAii;
		}
	}
}
