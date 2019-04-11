#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include <mpi.h>

/*
 * IF removed from init
 * IF _not_ removed from calc
 * collectives instead of p2p
 */

void SLGEWOPV_calc_last(double** A, double* b, double** T, double* x, int n, double* h, double* hh, int rank, int nprocs)
{
	// indexes
    int i,j,l;

    // map columns to process
    int* map;
    map=malloc(2*n*sizeof(int));
	for (i=0; i<2*n; i++)
	{
		map[i]= i % nprocs;
	}

    // num of cols per process
    int numcols;
    numcols=2*n/nprocs;

    // MPI derived types
	MPI_Datatype single_column;
	MPI_Type_vector (n, 1, 2*n, MPI_DOUBLE, & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/nprocs, 1, nprocs, MPI_DOUBLE, & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	MPI_Datatype interleaved_row_type;
	MPI_Type_create_resized (interleaved_row, 0, 1*sizeof(double), & interleaved_row_type);
	MPI_Type_commit (& interleaved_row_type);

	MPI_Datatype multiple_column;
	MPI_Type_vector (n * numcols, 1, nprocs , MPI_DOUBLE, & multiple_column );
	MPI_Type_commit (& multiple_column);

	MPI_Datatype multiple_column_type;
	MPI_Type_create_resized (multiple_column, 0, 1*sizeof(double), & multiple_column_type);
	MPI_Type_commit (& multiple_column_type);

	/*
	 *  init inhibition table
	 */

    MPI_Bcast (&b[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (i=0; i<n; i++)
	{
		x[i]=0.0;
	}

	if (rank==0)
	{
		for (j=0;j<n;j++)
		{
			for (i=0;i<n;i++)
			{
				T[i][j+n]=A[j][i]/A[i][i];
				T[i][j]=0;
			}
			T[j][j]=1/A[j][j];
		}
	}

	// scatter columns to nodes
	MPI_Scatter (&T[0][0], 1, multiple_column_type, &T[0][rank], 1, multiple_column_type, 0, MPI_COMM_WORLD);

	// broadcast of the last col of T (K part)
	MPI_Bcast (&T[0][n*2-1], 1, single_column, 0, MPI_COMM_WORLD);

	// broadcast of the last row of T (K part)
	MPI_Bcast (&T[n-1][n], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//TODO: combine previous broadcasts in a single one

	/*
	 *  calc inhibition sequence
	 */

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		for (i=l; i<=n-1; i++)
		{
			if(map[i]==rank)
			{
				x[i]=x[i]+T[l][i]*b[l];
			}
		}
		for (i=0; i<=l-1; i++)
		{
			b[i]=b[i]-T[l][n+i]*b[l];
			*h   =1/(1-T[i][n+l]*T[l][n+i]);
			*hh  =T[i][n+l]*(*h);

			/*
			if (rank==map[i])
			{
				T[i][i]=T[i][i]*(*h);
			}
			if (rank==map[l])
			{
				T[i][l]= -T[l][l]*(*hh);
			}
			*/

			// at every node: better calc than putting an IF ?
			T[i][i]=T[i][i]*(*h);
			T[i][l]= -T[l][l]*(*hh);
			//

			for (j=l+1; j<=n+l-1; j++)
			{
				if (rank==map[j])
				{
					T[i][j]=T[i][j]*(*h)-T[l][j]*(*hh);
				}
			}
		}

		// collect chunks of K to "future" last node
		MPI_Gather (&T[l-1][n+rank], 1, interleaved_row_type, &T[l-1][n], 1, interleaved_row_type, map[l-1], MPI_COMM_WORLD);

		//future last node broadcasts last row and col of K
		MPI_Bcast (&T[0][n+l-1], 1, single_column, map[l-1], MPI_COMM_WORLD);
		MPI_Bcast (&T[l-1][n], n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);

		//TODO: substitute Gather with an All-to-All
		//TODO: or combine broadcasts in a single one
	}

	// last level (l=0)
	for (i=0; i<=n-1; i++)
	{
		if (map[i]==rank)
		{
			x[i]=x[i]+T[0][i]*b[0];
		}
	}

	// collect solution
	MPI_Gather (&x[rank], 1, interleaved_row_type, &x[0], 1, interleaved_row_type, 0, MPI_COMM_WORLD);
}
