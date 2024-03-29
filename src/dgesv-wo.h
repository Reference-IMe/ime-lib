#ifndef __DGESV_WO_H__
#define __DGESV_WO_H__

#include "_gezr.h"
#include "dgeit-w_.h"
#include "helpers/matrix_basic.h"

/*
 *	solve (SV) system with general (GE) matrix A of doubles (D)
 *	of order n and with m r.h.s in matrix bb and solutions in xx
 *	with wide overwrite (WO) memory model
 *	with optimized initialization
 *
 */

void dgesv_wo (	int n,
				double** A,
				int m,
				double** bb,
				double** xx		)
{
    int i,j,l,rhs;
    int d=2*n;

    double h;
    double hh;
    double** T;

    T=AllocateMatrix2D_double (n, d, CONTIGUOUS);	// allocate table

    dgeit_w (A, T, n);		// init inhibition table
    dgezr (xx, n, m);		// init solution vectors

	for (l=n-1; l>=0; l--)
	{
		for (i=l; i<=n-1; i++)
		{
			for (rhs=0; rhs<m; rhs++)	// multiple rhs
			{
				xx[i][rhs]=xx[i][rhs]+T[l][i]*bb[l][rhs];
			}
		}
		for (i=0; i<=l-1; i++)
		{
			for (rhs=0; rhs<m; rhs++)	// multiple rhs
			{
				bb[i][rhs]=bb[i][rhs]-T[l][n+i]*bb[l][rhs];
			}
			h   =1/(1-T[i][n+l]*T[l][n+i]);
			hh  =T[i][n+l]*h;
			T[i][i]=T[i][i]*h;
			T[i][l]= -T[l][l]*hh;
			for (j=l+1; j<=n+l-1; j++)
			{
				T[i][j]=T[i][j]*h-T[l][j]*hh;
			}
		}
	}
	DeallocateMatrix2D_double (T, n, CONTIGUOUS);
}
