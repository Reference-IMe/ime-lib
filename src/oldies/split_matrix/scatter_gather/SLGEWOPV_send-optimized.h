#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include <mpi.h>

#include <unistd.h>


/*
 * IF removed from init
 * IF removed from calc
 * init with p2p
 * calc with collectives
 */
void SLGEWOPV_calc_sendopt(double** A, double* b, double* s, int n, double** K, double* H, int rank, int nprocs)
{
	int i,j,l;				// indexes
    int mycols;				// num of cols per process
    	mycols=n/nprocs;
    int myend;				// loop boundaries on local cols =mycols;
    int mystart;

	double** X=A;			// aliases
    double*  F=b;

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */

    double**	localK;
    			localK=AllocateMatrix2D(n,mycols,CONTIGUOUS);
    double**	localX;
    			localX=AllocateMatrix2D(n,mycols,CONTIGUOUS);

    double* 	lastKc;						// last col of K
    			lastKc=AllocateVector(n);
    double*		lastKr;						// last row of K
				lastKr=AllocateVector(n);
	double		tmpDiag;					// to hold a diagonal element during init

	/*
	 *  map columns to process
	 */

    int*	local;
    		local=malloc(n*sizeof(int));
	int*	map;
    		map=malloc(n*sizeof(int));
			for (i=0; i<n; i++)
			{
				map[i]= i % nprocs;			// who has the col i
				local[i]=floor(i/nprocs);	// position of the column i(global) in the local matrix

				s[i]=0.0;					// and init solution vector
			}
	int* 	global;
    		global=malloc(n*sizeof(int));
			for(i=0; i<mycols; i++)
			{
				global[i]= i * nprocs + rank; // position of the column i(local) in the global matrix
			}

	/*
	 * MPI derived types
	 */

	MPI_Datatype single_column;
	MPI_Type_vector (n, 1, n, MPI_DOUBLE, & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype local_single_column;
	MPI_Type_vector (n, 1, mycols, MPI_DOUBLE, & local_single_column );
	MPI_Type_commit (& local_single_column);

	MPI_Datatype multiple_column;
	MPI_Type_vector (n * mycols, 1, nprocs , MPI_DOUBLE, & multiple_column );
	MPI_Type_commit (& multiple_column);

	MPI_Datatype multiple_column_resized;
	MPI_Type_create_resized (multiple_column, 0, 1*sizeof(double), & multiple_column_resized);
	MPI_Type_commit (& multiple_column_resized);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/nprocs, 1, nprocs, MPI_DOUBLE, & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	MPI_Datatype interleaved_row_resized;
	MPI_Type_create_resized (interleaved_row, 0, 1*sizeof(double), & interleaved_row_resized);
	MPI_Type_commit (& interleaved_row_resized);


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
				K[i][j]=A[j][i]/A[i][i];
				// when i==j -> K[i][j]=1
			}

			MPI_Send (&K[0][j], 1, single_column, map[j], j+n, MPI_COMM_WORLD);
		}
		for (j=0; j<n; j++)
		{
			// init col of X
			tmpDiag=1/A[j][j];
			for (i=0; i<n; i++)
			{
				X[i][j]=0.0;
			}
			X[j][j]=tmpDiag;

			MPI_Send (&X[0][j], 1, single_column, map[j], j, MPI_COMM_WORLD);
		}
		// copy data into local buffer before broadcast
		for (i=0; i<n; i++)
		{
			lastKc[i]=K[i][n-1];
			lastKr[i]=K[n-1][i];
		}
	}

	// receive
	for (j=0; j<mycols; j++)
	{
		MPI_Recv (&localX[0][j], 1, local_single_column, 0, global[j], MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv (&localK[0][j], 1, local_single_column, 0, global[j]+n, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// broadcast last column and last row of K
	MPI_Bcast (&lastKc[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast (&lastKr[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//TODO: combine previous broadcasts in a single one

	/*
	 *  calc inhibition sequence
	 */

	for (l=n-1; l>0; l--)
	{

		for (i=0; i<l; i++)
		{
			H[i]=1/(1-lastKr[i]*lastKc[i]);
			F[i]=F[i]-F[l]*lastKr[i];
		}
		for (j=rank; j<l; j+=nprocs)
		{
			localX[j][local[j]]=H[j]*(localX[j][local[j]]-lastKc[j]*localX[l][local[j]]);
		}
		for (i=0; i<l; i++)
		{
			for (j=n-(nprocs-rank-1)-1; j>=l; j-=nprocs)
			{
				localX[i][local[j]]=H[i]*(localX[i][local[j]]-lastKc[i]*localX[l][local[j]]);
			}
			for (j=rank; j<l; j+=nprocs)
			{
				localK[i][local[j]]=H[i]*(localK[i][local[j]]-lastKc[i]*localK[l][local[j]]);
			}
		}

		// collect chunks of last row of K to "future" last node
		MPI_Gather (&localK[l-1][0], mycols, MPI_DOUBLE, &lastKr[0], 1, interleaved_row_resized, map[l-1], MPI_COMM_WORLD);

		//future last node broadcasts last row and col of K
		if (rank==map[l-1])
		{
			// copy data into local buffer before broadcast
			for (i=0; i<n; i++)
			{
				lastKc[i]=localK[i][local[l-1]];
			}
		}
		MPI_Bcast (&lastKc[0], n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
		MPI_Bcast (&lastKr[0], n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
		//TODO: combine previous broadcasts in a single one
	}

	// calc solution on nodes
	for (j=rank; j<n; j+=nprocs)
	{
		for (l=0; l<n; l++)
		{
			s[j]=F[l]*localX[l][local[j]]+s[j];
		}
	}

	// collect solution
	MPI_Gather (&s[rank], 1, interleaved_row, &s[0], 1, interleaved_row_resized, 0, MPI_COMM_WORLD);

	// cleanup
	free(local);
	free(global);
	free(map);
	DeallocateMatrix2D(localK,n,CONTIGUOUS);
	DeallocateMatrix2D(localX,n,CONTIGUOUS);
	DeallocateVector(lastKr);
	DeallocateVector(lastKc);
}
