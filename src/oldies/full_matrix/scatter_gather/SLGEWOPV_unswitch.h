#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include <mpi.h>


/*
 * IF _not_ removed from init
 * IF removed from calc
 * every node broadcasts chunks of last row
 * last node broadcast last col
 * master calcs and broadcasts auxiliary causes
 */
void SLGEWOPV_calc_unswitch(double** A, double* b, double* s, int n, double** K, double* H, int rank, int nprocs)
{
    double** X=A;
    double*  F=b;
    double tmpAdiag;

	// indexes
    int i,j,l;

    // map columns to process
    int* map;
    map=malloc(n*sizeof(int));
	for (i=0; i<n; i++)
	{
		map[i]= i % nprocs;
		s[i]=0.0;			// and init solution vector
	}

    // num of cols per process
    int numcols;
    numcols=n/nprocs;

    // MPI derived types
	MPI_Datatype single_column;
	MPI_Type_vector (n, 1, n, MPI_DOUBLE, & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (numcols, 1, nprocs, MPI_DOUBLE, & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	MPI_Datatype interleaved_row_type;
	MPI_Type_create_resized (interleaved_row, 0, 1*sizeof(double), & interleaved_row_type);
	MPI_Type_commit (& interleaved_row_type);

	MPI_Datatype multiple_column;
	MPI_Type_vector (n * numcols, 1, nprocs, MPI_DOUBLE, & multiple_column );
	MPI_Type_commit (& multiple_column);

	MPI_Datatype multiple_column_type;
	MPI_Type_create_resized (multiple_column, 0, 1*sizeof(double), &multiple_column_type);
	MPI_Type_commit (& multiple_column_type);


	/*
	 *  init inhibition table
	 */

	MPI_Bcast (&b[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank==0)
	{
		for (i=0; i<n; i++)
		{
			tmpAdiag=1/A[i][i];
			for (j=0; j<n; j++)
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
		}
	}

	// spread inhibition table
	MPI_Scatter (&X[0][0], 1, multiple_column_type, &X[0][rank], 1, multiple_column_type, 0, MPI_COMM_WORLD);
	MPI_Scatter (&K[0][0], 1, multiple_column_type, &K[0][rank], 1, multiple_column_type, 0, MPI_COMM_WORLD);

	// broadcast last column of K
	MPI_Bcast (&K[0][n-1], 1, single_column, 0, MPI_COMM_WORLD);

	// broadcast of the last row of K
	MPI_Bcast (&K[n-1][0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/*
	 *  calc inhibition sequence
	 */
	for (l=n-1; l>0; l--)
	{
		for (i=0; i<l; i++)
		{
			H[i]=1/(1-K[i][l]*K[l][i]);
		}
		for (j=rank; j<l; j+=nprocs)
		{
			X[j][j]=H[j]*(X[j][j]-K[j][l]*X[l][j]);
		}
		for (i=0; i<l; i++)
		{

			for (j=n-(nprocs-rank-1)-1; j>=l; j-=nprocs)
			{
				X[i][j]=H[i]*(X[i][j]-K[i][l]*X[l][j]);
			}
			for (j=rank; j<l; j+=nprocs)
			{
				K[i][j]=H[i]*(K[i][j]-K[i][l]*K[l][j]);
			}
		}

		// every node broadcasts its chunks of the last row of K
		MPI_Allgather (&K[l-1][rank], 1, interleaved_row_type, &K[l-1][0], 1, interleaved_row_type, MPI_COMM_WORLD);

		// broadcast last column of K
		MPI_Bcast (&K[0][l-1], 1, single_column, map[l-1], MPI_COMM_WORLD);
	}

	// calc auxiliary causes on root and then broadcast
    if (rank==0)
    {
		for (i=n-2; i>=0; i--)
		{
			for (l=i+1; l<n; l++)
			{
				F[i]=F[i]-F[l]*K[l][i];
			}
		}
    }
	MPI_Bcast (F, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// final solution on nodes
	for (j=rank; j<n; j+=nprocs)
	{
		for (l=0; l<n; l++)
		{
			s[j]=F[l]*X[l][j]+s[j];
		}
	}

	// collect solution
	MPI_Gather (&s[rank], 1, interleaved_row_type, &s[0], 1, interleaved_row_type, 0, MPI_COMM_WORLD);
}
