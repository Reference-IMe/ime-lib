#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include <mpi.h>


/*
 * IF _not_ removed from init
 * IF _not_ removed from calc
 * init with p2p
 * calc with collectives
 * master calcs and broadcasts auxiliary causes
 */
void SLGEWOPV_calc_sendopt(double** A, double* b, double* s, int n, double** K, double* H, int rank, int nprocs)
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

    // MPI derived types
	MPI_Datatype single_column;
	MPI_Type_vector (n, 1, n, MPI_DOUBLE, & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/nprocs, 1, nprocs, MPI_DOUBLE, & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	MPI_Datatype interleaved_row_type;
	MPI_Type_create_resized (interleaved_row, 0, 1*sizeof(double), & interleaved_row_type);
	MPI_Type_commit (& interleaved_row_type);


	/*
	 * init inhibition table
	 */
    MPI_Bcast (&b[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//send and recv inside loops, trying to overlap with calculations
	if (rank==0)
	{
		for (j=0; j<n; j++)
		{

			// init col of K
			for (i=0; i<n; i++)
			{
				if (i==j)
				{
					K[i][j]=1;
				}
				else
				{
					K[i][j]=A[j][i]/A[i][i];
				}
			}
			// send columns (all but last one) to single nodes (other than root)
			//last one of K is broadcasted to all nodes later
			if (map[j]!=0 && j<n-1)
			{
				MPI_Send (&K[0][j], 1, single_column, map[j], j+n, MPI_COMM_WORLD);
			}
		}
		for (j=0; j<n; j++)
		{
			// init col of X
			for (i=0; i<n; i++)
			{
				if (i==j)
				{
					X[i][i]=1/A[i][i];
				}
				else
				{
					X[i][j]=0.0;
				}
			}
			// send col of X to non-root nodes
			if (map[j]!=0)
			{
				MPI_Send (&X[0][j], 1, single_column, map[j], j, MPI_COMM_WORLD);
			}

		}
	}
	else // non root
	{
		for (j=0; j<n-1; j++)
		{
			// receive
			if (rank==map[j])
			{
				MPI_Recv (&X[0][j], 1, single_column, 0, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv (&K[0][j], 1, single_column, 0, j+n, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		if (rank==map[n-1])
		{
			MPI_Recv (&X[0][n-1], 1, single_column, 0, n-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}

	// broadcast last column of K
	MPI_Bcast (&K[0][n-1], 1, single_column, 0, MPI_COMM_WORLD);

	// broadcast of the last row of K
	MPI_Bcast (&K[n-1][0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//TODO: combine previous broadcasts in a single one

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
		/*
		if(rank!=map[l-1])
		{
			MPI_Send (&K[l-1][rank],1,interleaved_row, map[l-1],rank, MPI_COMM_WORLD);
		}
		else
		{
			for(j=0;j<nprocs;j++)
			{
				if(j!=rank)
				{
					MPI_Recv (&K[l-1][j],1,interleaved_row, j,j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
			}
		}
		*/
		// collect chunks of K to "future" last node
		MPI_Gather (&K[l-1][rank], 1, interleaved_row_type, &K[l-1][0], 1, interleaved_row_type, map[l-1], MPI_COMM_WORLD);

		//future last node broadcasts last row and col of K
		MPI_Bcast (&K[0][l-1], 1, single_column, map[l-1], MPI_COMM_WORLD);
		MPI_Bcast (&K[l-1][0], n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);

		//TODO: combine previous broadcasts in a single one
	}

	// master calcs and broadcasts auxiliary causes
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

	// calc solution on nodes
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
