#include <mpi.h>
#include "helpers/macros.h"

/*
 * 	distributed
 *	init (IT) inhibition table T
 *	from general (GE) system matrix A of doubles (D)
 *
 */

#ifndef __pbDGEIT_CX_H__
#define __pbDGEIT_CX_H__

void pbDGEIT_CX(double** A, double** Tlocal, double** lastK, int n, int bf, MPI_Comm comm, int rank, MPI_Comm comm_row, int rank_col_in_row, MPI_Comm comm_col, int rank_row_in_col, int cprocs)
{
    double** vTlocal;

	// last rows and cols of K
	double** vlastK;

    int cprocrows = sqrt(cprocs);
    int cproccols = cprocrows;
    int myrows   = n/cprocrows;		// num of rows per process
    int mycols   = n/cproccols;		// num of cols per process

	MPI_Datatype lastK_chunk;
	MPI_Type_vector (bf, mycols, n, MPI_DOUBLE, &lastK_chunk);
	MPI_Type_commit (&lastK_chunk);

	MPI_Datatype lastK_chunk_resized;
	MPI_Type_create_resized (lastK_chunk, 0, mycols*sizeof(double), &lastK_chunk_resized);
	MPI_Type_commit (&lastK_chunk_resized);

	int i,j;
	if (rank_row_in_col==0)	// head of the column of procs
	{
		vTlocal=AllocateMatrix2D(n, mycols, CONTIGUOUS);
		vlastK=AllocateMatrix2D(2*bf, n, CONTIGUOUS);	// last rows [0 - (bf-1)] and cols [ bf - (2bf -1)] of K
		pvDGEIT_CX(A, vTlocal, vlastK, n, bf, comm_row, rank_col_in_row, cproccols);

		/*
		for (i=0; i<cproccols; i++)
		{
			MPI_Barrier(comm_row);
			if (rank_col_in_row==i)
			{
				printf("vTlocal and vlastK in %d (%d,%d):\n",rank,rank_row_in_col,rank_col_in_row);
				PrintMatrix2D(vTlocal, n, mycols);
				printf("\n");
				PrintMatrix2D(vlastK, 2*bf, n);
				printf("\n");
			    fflush(stdout);
			}
		}
		MPI_Barrier(comm_row);
		*/

		MPI_Scatter (&vTlocal[0][0], myrows*mycols, MPI_DOUBLE, &Tlocal[0][0], myrows*mycols, MPI_DOUBLE, 0, comm_col);
		for (i=0; i<bf; i++)
		{
			for (j=0; j<mycols; j++)
			{
				lastK[i][j]=vlastK[i][rank_col_in_row*mycols+j];
			}
		}
		MPI_Bcast ( &lastK[0][0], bf*mycols, MPI_DOUBLE, 0, comm_col);
		MPI_Scatter ( &vlastK[bf][0], 1, lastK_chunk_resized, &lastK[bf][0], bf*mycols, MPI_DOUBLE, 0, comm_col);
	}
	else
	{
		vTlocal=NULL;
		vlastK=NULL;
		MPI_Scatter (NULL, myrows*mycols, MPI_DOUBLE, &Tlocal[0][0], myrows*mycols, MPI_DOUBLE, 0, comm_col);
		MPI_Bcast   ( &lastK[0][0], bf*mycols, MPI_DOUBLE, 0, comm_col);
		MPI_Scatter (NULL, 1, lastK_chunk_resized, &lastK[bf][0], bf*mycols, MPI_DOUBLE, 0, comm_col);
	}
	DeallocateMatrix2D(vlastK,2*bf,CONTIGUOUS);
	DeallocateMatrix2D(vTlocal,n,CONTIGUOUS);
}

#endif
