#include <mpi.h>
#include "helpers/macros.h"

/*
 * 	distributed
 *	init (IT) inhibition table T
 *	from general (GE) system matrix A of doubles (D)
 *
 */

#ifndef __pvDGEIT_CX_IND_H__
#define __pvDGEIT_CX_IND_H__

void pvDGEIT_CX_ind(double** A, double** Tlocal, double** lastK, int n, int bf, MPI_Comm comm, int rank, int cprocs, int* map, int* global, int* local)
{
	int i,j;

    int myArows;					// num of A rows/cols per process
    	myArows=n/cprocs;

    int myAchunks = myArows;					// num of A chunks per process

    int myKcols;
    	myKcols=n/cprocs;

    /*
	double*  lastKc=&lastK[0][0];						// alias for last col
	double*  lastKr=&lastK[1][0];						// alias for last row
	*/

	double** lastKr;
				lastKr=malloc(bf*sizeof(double*));
				for(i=0;i<bf;i++)
				{
					lastKr[i]=lastK[i];						// alias for last row
				}
	double** lastKc;
				lastKc=malloc(bf*sizeof(double*));
				for(i=0;i<bf;i++)
				{
					lastKc[i]=lastK[bf+i];						// alias for last col
				}


	// rows of A to be extracted and sent
	MPI_Datatype A_rows_interleaved;
	MPI_Type_vector (myAchunks, bf*n, bf*n*cprocs , MPI_DOUBLE, & A_rows_interleaved );
	MPI_Type_commit (& A_rows_interleaved);

	// rows of A to be extracted and sent, properly resized for scattering
	MPI_Datatype A_rows_interleaved_resized;
	MPI_Type_create_resized (A_rows_interleaved, 0, bf*n*sizeof(double), & A_rows_interleaved_resized);
	MPI_Type_commit (& A_rows_interleaved_resized);

	// rows of A extracted, to be stored as contiguous columns in T (K part)
	MPI_Datatype K_column_contiguous;
	MPI_Type_vector (n, 1, myKcols, MPI_DOUBLE, & K_column_contiguous );
	MPI_Type_commit (& K_column_contiguous);

	// rows of A extracted, to be stored as contiguous columns in T (K part), properly resized for scattering
	MPI_Datatype K_column_contiguous_resized;
	MPI_Type_create_resized (K_column_contiguous, 0, 1*sizeof(double), & K_column_contiguous_resized);
	MPI_Type_commit (& K_column_contiguous_resized);


    // prepare entire last rows of K and the entire diagonal of A in buffer to be sent
    if (rank==0)
    {
		for (i=0;i<n;i++)
		{//reuse memory
			for (j=0;j<bf;j++)
			{
				// store the last rows of K in lastKr in the same order
				// (lastKr[0] is the first one of the block of last rows of K)
				lastKr[j][i]=A[i][n-bf+j]/A[n-bf+j][n-bf+j]; // last cols of A -> last rows of K
			}
			// store the diagonal of A at the beginning of lastKc because it's contiguous to lastKr
			lastKc[0][i]=A[i][i]; // diagonal of A
		}
    }

    MPI_Bcast (&lastK[0][0], n*(bf+1), MPI_DOUBLE, 0, comm); // last cols and diagonal of A
	MPI_Scatter (&A[0][0], n*myArows, MPI_DOUBLE, &Tlocal[0][0], myArows, K_column_contiguous_resized, 0, comm);	// scatter columns to nodes

    // init
	for (i=0;i<n;i++)
	{
		for (j=0;j<myArows;j++)
		{
			// X part
			if (i==global[j])
			{
				Tlocal[i][j]=1/lastKc[0][i];
			}
			else
			{
				//Xlocal[i][j]=0;
				Tlocal[i][j]=Tlocal[i][j]/lastKc[0][i];
			}
			// K part
			//Klocal[i][j]=Klocal[i][j]/lastKc[0][i];
		}
	}

	// prepare (copy into local buffer) last cols of K
	if (rank==map[n-1]) // the process holding the last column holds the last bf columns [n-bf..n-1]
	{
		for (i=0; i<n; i++)
		{
			for (j=0;j<bf;j++)
			{
				// store the last bf cols of K in lastKc in the same order, but transposed (columns stored in rows)
				// (lastKc[0] is the first one of the block of last cols of K)
				lastKc[j][i]=Tlocal[i][local[n-bf]+j];
			}
		}
	}

	MPI_Bcast (&lastKc[0][0], n*bf, MPI_DOUBLE, map[n-1], comm);	// broadcast of the last cols of K

	NULLFREE(lastKc);
	NULLFREE(lastKr);

}

#endif
