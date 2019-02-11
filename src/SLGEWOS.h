#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers/selfie.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"

void SLGEWOS_calc_last(double** A, double* b, double** T, double* x, int n, double* h, double* hh)
{
    int i,j,l;
    //int rows=n;
    //int cols=n;

    //double h,hh;

	for (i=0;i<n;i++)
	{
		for (j=0;j<n;j++)
		{
			T[i][j+n]=A[j][i]/A[i][i];
			T[i][j]=0;
		}
		T[i][i]=1/A[i][i];
		x[i]=0.0;
	}

	for (l=n-1; l>=0; l--)
	{
		for (i=l; i<=n-1; i++)
		{
			x[i]=x[i]+T[l][i]*b[l];
		}
		for (i=0; i<=l-1; i++)
		{
			b[i]=b[i]-T[l][n+i]*b[l];
			*h   =1/(1-T[i][n+l]*T[l][n+i]);
			*hh  =T[i][n+l]*(*h);
			T[i][i]=T[i][i]*(*h);
			//printf("%d,%d\t",i,i);
			//T[i][l]= -T[l][l]*T[i][n+l]*H[i];
			T[i][l]= -T[l][l]*(*hh);
			for (j=l+1; j<=n+l-1; j++)
			{
				//printf("%d,%d\t",i,j);
				//T[i][j]=(T[i][j]-T[l][j]*T[i][n+l])*H[i];
				T[i][j]=T[i][j]*(*h)-T[l][j]*(*hh);
			}
			//printf("\n");
		}
	}
}

/*
void SLGEWOS_calc_last(double** A, double* b, double** T, double* x, int n, double* H)
{
    int i,j,l;
    int rows=n;
    int cols=n;

	for (i=0;i<rows;i++)
	{
		for (j=0;j<cols;j++)
		{
			if (i==j)
			{
				T[i][j]=1/A[i][j];
				T[i][j+n]=1;
			}
			else
			{
				T[i][j]=0;
				T[i][j+n]=A[j][i]/A[i][i];
			}
		}
		x[i]=0.0;
	}

	for (l=n-1; l>=0; l--)
	{
		for (i=l; i<=n-1; i++)
		{
			x[i]=x[i]+T[l][i]*b[l];
		}
		for (i=0; i<=l-1; i++)
		{
			b[i]=b[i]-T[l][n+i]*b[l];
			H[i]=1/(1-T[i][n+l]*T[l][n+i]);
			T[i][i]=T[i][i]*H[i];
			//printf("%d,%d\t",i,i);
			T[i][l]= -T[l][l]*T[i][n+l]*H[i];
			for (j=l+1; j<=n+l-1; j++)
			{
				//printf("%d,%d\t",i,j);
				T[i][j]=(T[i][j]-T[l][j]*T[i][n+l])*H[i];
			}
			//printf("\n");
		}
	}
}
*/

void SLGEWOS_calc_paper(double** A, double* b, double** T, double* x, int n, double* H)
{
    int i,j,l;
    int rows=n;
    int cols=n;

	for (i=0;i<rows;i++)
	{
		for (j=0;j<cols;j++)
		{
			if (i==j)
			{
				T[i][j]=1/A[i][j];
				T[i][j+n]=1;
			}
			else
			{
				T[i][j]=0;
				T[i][j+n]=A[j][i]/A[i][i];
			}
		}
		x[i]=0.0;
	}

	for (l=n-1; l>=0; l--)
	{
		for (i=l; i<=n-1; i++)
		{
			x[i]=x[i]+T[l][i]*b[l];
		}
		for (i=0; i<=l-1; i++)
		{
			H[i]=1/(1-T[i][l+n]*T[l][i+n]);
			b[i]=b[i]-T[l][i+n]*b[l];
			for (j=0; j<=n-1; j++)
			{
				if (j<l && j!=i)
				{
					T[i][j+n]=(T[i][j+n]-T[l][j+n]*T[i][n+l])*H[i];
				}
				else
				{
					T[i][j]=(T[i][j]-T[l][j]*T[i][n+l])*H[i];
				}
			}
		}
	}
}

void SLGEWOS_calc(double** A, double* b, double* s, int n, double** K, double* H, double* F)
{
    int i,j,l;
    int rows=n;
    int cols=n;
    double** X=A;
    double tmpAdiag;

	for (i=0;i<rows;i++)
	{
		tmpAdiag=1/A[i][i];
		for (j=0;j<cols;j++)
		{
			if (i==j)
			{
				X[i][j]=tmpAdiag;
				//K[i][j]=1;
			}
			else
			{
				K[i][j]=A[j][i]*tmpAdiag;

				// ATTENTION : transposed
				X[j][i]=0.0;

			}
		}
		F[i]=b[i];
		s[i]=0.0;
	}

	for (l=rows-1; l>0; l--)
	{
		for (i=0; i<l; i++)
		{
			H[i]=1/(1-K[i][l]*K[l][i]);
			for (j=0; j<cols; j++)
			{
				X[i][j]=H[i]*(X[i][j]-K[i][l]*X[l][j]);
			}
			for (j=0; j<l; j++)
			{
				if (j!=i)
				{
					K[i][j]=H[i]*(K[i][j]-K[i][l]*K[l][j]);
				}
			}
		}
	}

	for (i=rows-2; i>=0; i--)
	{
		for (l=i+1; l<rows; l++)
		{
			F[i]=F[i]-F[l]*K[l][i];
		}
	}

	for (j=0; j<cols; j++)
	{
		for (l=0; l<rows; l++)
		{
			s[j]=F[l]*X[l][j]+s[j];
		}
	}
}

void SLGEWOS_calc_unwind(double** A, double* b, double* s, int n, double** K, double* H, double* F)
{
    int i,j,l;
    int rows=n;
    int cols=n;
    double** X=A;
    double tmpAdiag;

	for (i=0; i<rows; i++)
	{
		tmpAdiag=1/A[i][i];
		for (j=0; j<cols; j++)
		{
			if (i==j)
			{
				X[i][j]=tmpAdiag;
				K[i][j]=1;
			}
			else
			{
				K[i][j]=A[j][i]*tmpAdiag;

				// ATTENTION : transposed
				X[j][i]=0.0;

			}
		}
		F[i]=b[i];
		s[i]=0.0;
	}

	for (l=rows-1; l>0; l--)
	{
		for (i=0; i<l; i++)
		{
			H[i]=1/(1-K[i][l]*K[l][i]);
			X[i][i]=H[i]*(X[i][i]);
			for (j=l; j<cols; j++)
			{
				X[i][j]=H[i]*(X[i][j]-K[i][l]*X[l][j]);
			}

		}
		for (i=0; i<l; i++)
		{
			for (j=0; j<l; j++)
			{
				K[i][j]=H[i]*(K[i][j]-K[i][l]*K[l][j]);
			}
		}
	}

	for (i=rows-2; i>=0; i--)
	{
		for (l=i+1; l<rows; l++)
		{
			F[i]=F[i]-F[l]*K[l][i];
		}
	}

	for (j=0; j<cols; j++)
	{
		for (l=0; l<rows; l++)
		{
			s[j]=F[l]*X[l][j]+s[j];
		}
	}
}

void SLGEWOS_calc_allocX(double** A, double* b, double* s, int n, double** X, double** K, double* H, double* F)
{
    int i,j,l;
    int rows=n;
    int cols=n;

	for (i=0;i<rows;i++)
	{
		for (j=0;j<cols;j++)
		{
			if (i==j)
			{
				X[i][j]=1/A[i][j];
				//K[i][j]=1;
			}
			else
			{
				X[i][j]=0.0;
				K[i][j]=A[j][i]/A[i][i];
			}
		}
		F[i]=b[i];
		s[i]=0.0;
	}

	for (l=rows-1; l>0; l--)
	{
		for (i=0; i<l; i++)
		{
			H[i]=1/(1-K[i][l]*K[l][i]);
			for (j=0; j<cols; j++)
			{
				X[i][j]=H[i]*(X[i][j]-K[i][l]*X[l][j]);
			}
			for (j=0; j<l; j++)
			{
				if (j!=i)
				{
					K[i][j]=H[i]*(K[i][j]-K[i][l]*K[l][j]);
				}
			}
		}
	}

	for (i=rows-2; i>=0; i--)
	{
		for (l=i+1; l<rows; l++)
		{
			F[i]=F[i]-F[l]*K[l][i];
		}
	}

	for (j=0; j<cols; j++)
	{
		for (l=0; l<rows; l++)
		{
			s[j]=F[l]*X[l][j]+s[j];
		}
	}
}

void SLGEWOS(double** A, double* b, double* s, int n)
{
    //double** X;
    double** K;
    double*  H;
    double*  F;

    //X=AllocateMatrix2D(n,n,CONTIGUOUS);
    K=AllocateMatrix2D(n,n,CONTIGUOUS);

    H=AllocateVector(n);
    F=AllocateVector(n);

    SLGEWOS_calc(A, b, s, n, K, H, F);
    //SLGEWOS_calc_allocX(A, b, s, n, X, K, H, F);

    //DeallocateMatrix2D(X,n,CONTIGUOUS);
    DeallocateMatrix2D(K,n,CONTIGUOUS);

    DeallocateVector(H);
    DeallocateVector(F);
}
