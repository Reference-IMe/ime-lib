#include "helpers/matrix.h"
#include "DGEIT_W_1D.h"
#include "DGEZR_1D.h"

/*
 *	with optimized initialization, 1D version
 */

void DGESV_WO_1D(int n, double* A, int m, double* bb, double* xx)
{
    int i,j,l,rhs;
    int d=2*n;

    double h;
    double hh;
    double denAii;
    double* T;


    T=AllocateMatrix1D(n,d);	// allocate table

    DGEIT_W_1D(A, T, n);		// init inhibition table
    DGEZR_1D(xx, n, m);			// init solution vectors

	for (l=n-1; l>=0; l--)
	{
		for (i=l; i<=n-1; i++)
		{
			for (rhs=0; rhs<m; rhs++)	// multiple rhs
			{
				xx[i*m+rhs]=xx[i*m+rhs]+T[l*d+i]*bb[l*m+rhs];
			}
		}
		for (i=0; i<=l-1; i++)
		{
			for (rhs=0; rhs<m; rhs++)	// multiple rhs
			{
				bb[i*m+rhs]=bb[i*m+rhs]-T[l*d+n+i]*bb[l*m+rhs];
			}
			h   =1/(1-T[i*d+n+l]*T[l*d+n+i]);
			hh  =T[i*d+n+l]*h;
			T[i*d+i]=T[i*d+i]*h;
			T[i*d+l]= -T[l*d+l]*hh;
			for (j=l+1; j<=n+l-1; j++)
			{
				T[i*d+j]=T[i*d+j]*h-T[l*d+j]*hh;
			}
		}
	}
	DeallocateMatrix1D(T);
}
