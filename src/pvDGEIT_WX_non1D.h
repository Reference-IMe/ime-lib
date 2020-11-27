#include <mpi.h>

/*
 * 	distributed
 *	init (IT) inhibition table T
 *	from general (GE) system matrix A of doubles (D)
 *
 */

#ifndef __pvDGEIT_WX_H__
#define __pvDGEIT_WX_H__

void pvDGEIT_WX(double** A, double** Xlocal, double** Klocal, double** lastK, int n, MPI_Comm comm, int rank, int cprocs, int* map, int* global, int* local)
{
	int i,j;

    int myAchunks;					// num of A rows/cols per process
    	myAchunks=n/cprocs;

    int myKcols;
    	myKcols=n/cprocs;

	double*  lastKc=&lastK[0][0];						// alias for last col
	double*  lastKr=&lastK[1][0];						// alias for last row

	// rows of A to be extracted and sent
	MPI_Datatype A_rows_interleaved;
	MPI_Type_vector (myAchunks, n, n*cprocs , MPI_DOUBLE, & A_rows_interleaved );
	MPI_Type_commit (& A_rows_interleaved);

	// rows of A to be extracted and sent, properly resized for scattering
	MPI_Datatype A_rows_interleaved_resized;
	MPI_Type_create_resized (A_rows_interleaved, 0, n*sizeof(double), & A_rows_interleaved_resized);
	MPI_Type_commit (& A_rows_interleaved_resized);

	// rows of A extracted, to be stored as contiguous columns in T (K part)
	MPI_Datatype K_column_contiguous;
	MPI_Type_vector (n, 1, myKcols, MPI_DOUBLE, & K_column_contiguous );
	MPI_Type_commit (& K_column_contiguous);

	// rows of A extracted, to be stored as contiguous columns in T (K part), properly resized for scattering
	MPI_Datatype K_column_contiguous_resized;
	MPI_Type_create_resized (K_column_contiguous, 0, 1*sizeof(double), & K_column_contiguous_resized);
	MPI_Type_commit (& K_column_contiguous_resized);


    // prepare entire last row of K and entire diagonal of A in buffer to be sent
    if (rank==0)
    {
		for (i=0;i<n;i++)
		{//reuse memory
			lastKr[i]=A[i][n-1]/A[n-1][n-1]; // last col of A -> last row of K
			lastKc[i]=A[i][i]; // diagonal
		}
    }

    MPI_Bcast (&lastK[0][0], 2*n, MPI_DOUBLE, 0, comm); // last col and diagonal of A
	MPI_Scatter (&A[0][0], 1, A_rows_interleaved_resized, &Klocal[0][0], myAchunks, K_column_contiguous_resized, 0, comm);	// scatter columns to nodes

    // init
	for (i=0;i<n;i++)
	{
		for (j=0;j<myAchunks;j++)
		{
			// X part
			if (i==global[j])
			{
				Xlocal[i][j]=1/lastKc[i];
			}
			else
			{
				Xlocal[i][j]=0;
			}
			// K part
			Klocal[i][j]=Klocal[i][j]/lastKc[i];
		}
	}

	// prepare (copy into local buffer) last col of T (K part)
	if (rank==map[n-1])
	{
		for (i=0; i<n; i++)
		{
			lastKc[i]=Klocal[i][local[n-1]];
		}
	}

	MPI_Bcast (&lastK[0][0], n, MPI_DOUBLE, map[n-1], comm);	// broadcast of the last col of T (K part)

}

#endif
