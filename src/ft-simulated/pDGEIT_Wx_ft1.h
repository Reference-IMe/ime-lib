#include <mpi.h>
#include "../helpers/matrix.h"

/*
 * 	distributed
 *	init (IT) inhibition table T
 *	from general (GE) system matrix A of doubles (D)
 *
 */

#ifndef __pDGEIT_W_FT_H__
#define __pDGEIT_W_FT_H__

void pDGEIT_W_ft1(double** A, double** Tlocal, double** TlastK, int n, int rank, int cprocs, int sprocs, int* map, int* global, int* local, int failing_rank)
{
	int i,j;

    int myAchunks;					// num of A rows/cols per process
    	myAchunks=n/cprocs;

    int Tcols=2*n;
    int myTcols=Tcols/cprocs;

	double*  TlastKc=&TlastK[0][0];						// alias for last col
	double*  TlastKr=&TlastK[1][0];						// alias for last row

	double** Slocal;
	Slocal=AllocateMatrix2D(n,myTcols, CONTIGUOUS);

	// rows of A to be extracted and sent
	MPI_Datatype A_rows_interleaved;
	MPI_Type_vector (myAchunks, n, n*cprocs , MPI_DOUBLE, & A_rows_interleaved );
	MPI_Type_commit (& A_rows_interleaved);

	// rows of A to be extracted and sent, properly resized for scattering
	MPI_Datatype A_rows_interleaved_resized;
	MPI_Type_create_resized (A_rows_interleaved, 0, n*sizeof(double), & A_rows_interleaved_resized);
	MPI_Type_commit (& A_rows_interleaved_resized);

	// rows of A extracted, to be stored as contiguous columns in T (K part)
	MPI_Datatype KinT_column_contiguous;
	MPI_Type_vector (n, 1, myTcols, MPI_DOUBLE, & KinT_column_contiguous );
	MPI_Type_commit (& KinT_column_contiguous);

	// rows of A extracted, to be stored as contiguous columns in T (K part), properly resized for scattering
	MPI_Datatype KinT_column_contiguous_resized;
	MPI_Type_create_resized (KinT_column_contiguous, 0, 1*sizeof(double), & KinT_column_contiguous_resized);
	MPI_Type_commit (& KinT_column_contiguous_resized);

    // prepare entire last row of K and entire diagonal of A in buffer to be sent
    if (rank==0)
    {
		for (i=0;i<n;i++)
		{//reuse memory
			TlastKr[i]=A[i][n-1]/A[n-1][n-1]; // last col of A -> last row of K
			TlastKc[i]=A[i][i]; // diagonal
		}
    }

    MPI_Bcast (&TlastK[0][0], Tcols, MPI_DOUBLE, 0, MPI_COMM_WORLD); // last col and diagonal of A
	MPI_Scatter (&A[0][0], 1, A_rows_interleaved_resized, &Tlocal[0][myAchunks], myAchunks, KinT_column_contiguous_resized, 0, MPI_COMM_WORLD);	// scatter columns to nodes

	if (rank<cprocs)
	{
		// init for calc. nodes
		for (i=0;i<n;i++)
		{
			for (j=0;j<myAchunks;j++)
			{
				// X part
				if (i==global[j])
				{
					Tlocal[i][j]=1/TlastKc[i];
				}
				else
				{
					Tlocal[i][j]=0;
				}
				// K part
				Tlocal[i][myAchunks+j]=Tlocal[i][myAchunks+j]/TlastKc[i];
			}
		}

	}
	else // rank=cprocs (last rank -> checksumming)
	{
		// init for checksumming nodes
		for (i=0;i<n;i++)
		{
			for (j=0;j<myTcols;j++)
			{
				Tlocal[i][j]=0;
			}
		}
	}// init done


	if (sprocs>0) // in ft mode
	{
		MPI_Reduce(&Tlocal[0][0], &Slocal[0][0], n*myTcols, MPI_DOUBLE, MPI_SUM, cprocs, MPI_COMM_WORLD);

		/*
		// fake fault recovery init
		if (rank==failing_rank)
		{
			MPI_Send(&Tlocal[0][0],n*myTcols,MPI_DOUBLE,cprocs,cprocs,MPI_COMM_WORLD);
		}
		if (rank==cprocs)
		{
			MPI_Recv(&Slocal[0][0],n*myTcols,MPI_DOUBLE,failing_rank,cprocs,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		*/

		if (rank==cprocs) // on (single) checksumming node
		{
			//Tlocal=Slocal
			for (i=0;i<n;i++)
			{
				for (j=0;j<myTcols;j++)
				{
					Tlocal[i][j]=Slocal[i][j];
				}
			}
			/*
			printf("\n\n Matrix S:\n");
			PrintMatrix2D(Tlocal, n, myTcols);
			*/
		}
	}
	DeallocateMatrix2D(Slocal,n,CONTIGUOUS);

	/*
	MPI_Barrier(MPI_COMM_WORLD);
	sleep(2*rank);
	printf("\n\n Matrix Tloc (%d):\n",rank);
	PrintMatrix2D(Tlocal, n, myTcols);
	MPI_Barrier(MPI_COMM_WORLD);
	*/

	// prepare (copy into local buffer) last col of T (K part)
	if (rank==map[n-1])
	{
		for (i=0; i<n; i++)
		{
			TlastKc[i]=Tlocal[i][local[2*n-1]];
		}
	}

	MPI_Bcast (&TlastK[0][0], n, MPI_DOUBLE, map[n-1], MPI_COMM_WORLD);	// broadcast of the last col of T (K part)

	MPI_Type_free(&A_rows_interleaved);
	MPI_Type_free(&A_rows_interleaved_resized);
	MPI_Type_free(&KinT_column_contiguous);
	MPI_Type_free(&KinT_column_contiguous_resized);
}

#endif
