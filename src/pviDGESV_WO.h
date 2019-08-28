#include <mpi.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "DGEZR.h"
#include "DGEIT_Wx.h"

/*
 *	solve (SV) system with general (GE) matrix A of doubles (D)
 *	of order n and with m r.h.s in matrix bb[n,m] and solutions in xx[n,m]
 *	with wide overwrite (WO) memory model
 *	parallelized in interleaved columns (pvi) over cprocs calculating processors
 *	without optimized initialization
 *
 */

void pviDGESV_WO(int n, double** A, int m, double** bb, double** xx, int rank, int cprocs)
{
	int i,j,l;						// indexes
    int Tcols=2*n;					// total num of cols (X + K)
    int myTcols;					// num of T cols per process
    	myTcols=Tcols/cprocs;
    int myAchunks;					// num of A rows/cols per process
    	myAchunks=n/cprocs;
    int myxxrows=myAchunks;
    int myend;						// loop boundaries on local cols =myTcols/2;
    int mystart;
    int rhs;

    int myAcols=n/cprocs;
    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Tlocal;
    		 Tlocal=AllocateMatrix2D(n, myTcols, CONTIGUOUS);

	double** TlastK;
			 TlastK=AllocateMatrix2D(2,n, CONTIGUOUS);	// last col [0] and row [1] of T (K part)
	double*  TlastKc=&TlastK[0][0];						// alias for last col
	double*  TlastKr=&TlastK[1][0];						// alias for last row

    double* h;							// helper vectors
    		h=AllocateVector(n);
    double* hh;
			hh=AllocateVector(n);

	/*
	 * map columns to process
	 */
	int*	local;
    		local=malloc(Tcols*sizeof(int));
    int*	map;
    		map=malloc(Tcols*sizeof(int));
			for (i=0; i<Tcols; i++)
			{
				map[i]= i % cprocs;			// who has the col i
				local[i]=floor(i/cprocs);	// position of the column i(global) in the local matrix
			}
    int*	global;
    		global=malloc(myTcols*sizeof(int));
			for (i=0; i<myTcols; i++)
			{
				global[i]= i * cprocs + rank; // position of the column i(local) in the global matrix
			}

    /*
     * MPI derived types
     */

	MPI_Datatype TlastKr_chunks;
	MPI_Type_vector (n/cprocs, 1, cprocs, MPI_DOUBLE, & TlastKr_chunks );
	MPI_Type_commit (& TlastKr_chunks);

	MPI_Datatype TlastKr_chunks_resized;
	MPI_Type_create_resized (TlastKr_chunks, 0, 1*sizeof(double), & TlastKr_chunks_resized);
	MPI_Type_commit (& TlastKr_chunks_resized);

	// rows of xx to be extracted
	MPI_Datatype xx_rows_interleaved;
	MPI_Type_vector (myxxrows, m, m*cprocs, MPI_DOUBLE, & xx_rows_interleaved );
	MPI_Type_commit (& xx_rows_interleaved);

	// rows of xx to be extracted, properly resized for gathering
	MPI_Datatype xx_rows_interleaved_resized;
	MPI_Type_create_resized (xx_rows_interleaved, 0, m*sizeof(double), & xx_rows_interleaved_resized);
	MPI_Type_commit (& xx_rows_interleaved_resized);

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


	/*
	 *  init inhibition table
	 */

	// send all r.h.s to all procs
    MPI_Bcast (&bb[0][0], n*m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // init (zero) solution vectors
    DGEZR(xx, n, m);

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

    // init
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

	// prepare (copy into local buffer) last col of T (K part)
	if (rank==map[n-1])
	{
		for (i=0; i<n; i++)
		{
			TlastKc[i]=Tlocal[i][local[2*n-1]];
		}
	}

	MPI_Bcast (&TlastK[0][0], n, MPI_DOUBLE, map[n-1], MPI_COMM_WORLD);	// broadcast of the last col of T (K part)

	/*
	 *  calc inhibition sequence
	 */

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		// ALL procs
		// update solutions
		// l .. n-1
		mystart=local[l];
		if (rank<map[l])
		{
			mystart++;
		}
		for (i=mystart; i<=local[n-1]; i++)
		{
			for (rhs=0;rhs<m;rhs++)
			{
				xx[global[i]][rhs]=xx[global[i]][rhs]+Tlocal[l][i]*bb[l][rhs];
			}
		}

		// ALL procs
		// update helpers
		for (i=0; i<=l-1; i++)
		{
			h[i]   = 1/(1-TlastKc[i]*TlastKr[i]);
			hh[i]  = TlastKc[i]*h[i];
			for (rhs=0;rhs<m;rhs++)
			{
				bb[i][rhs] = bb[i][rhs]-TlastKr[i]*bb[l][rhs];
			}
		}

		// update T
		// to avoid IFs: each process loops on its own set of cols, with indirect addressing

		// 0 .. l-1
		// ALL procs
		// processes with diagonal elements not null
		mystart=0;
		myend=local[l-1];
		if (rank>map[l-1])
		{
			myend--;
		}
		for (i=mystart; i<=myend; i++)
		{
			Tlocal[global[i]][i]=Tlocal[global[i]][i]*h[global[i]];
		}

		// l
		// ONLY one proc
		// process with full not null column (column l)
		if (rank==map[l])
		{
			for (i=0; i<=l-1; i++)
			{
				Tlocal[i][local[l]]= -Tlocal[l][local[l]]*hh[i];
			}
		}// TODO: incorporate with next block to form l .. n+l-1

		// l+1 .. n+l-1
		// ALL procs
		// all other cases
		mystart=local[l+1]; // TODO: incorporate l
		if (rank<map[l+1])
		{
			mystart++;
		}
		myend=local[n+l-1];
		if (rank>map[n+l-1])
		{
			myend--;
		}
		for (i=0; i<=l-1; i++)
		{
			for (j=mystart; j<=myend; j++)
			{
				Tlocal[i][j]=Tlocal[i][j]*h[i]-Tlocal[l][j]*hh[i];
			}
		}

		// collect chunks of last row of K to "future" last node
		MPI_Gather (&Tlocal[l-1][local[n]], myTcols/2, MPI_DOUBLE, &TlastKr[0], 1, TlastKr_chunks_resized, map[l-1], MPI_COMM_WORLD);

		//future last node broadcasts last row and col of K
		if (rank==map[l-1])
		{
			// copy data into local buffer before broadcast
			for (i=0; i<n; i++)
			{
				TlastKc[i]=Tlocal[i][local[n+l-1]];
			}
		}
		/*
		MPI_Bcast (&TlastKc[0], n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
		MPI_Bcast (&TlastKr[0], n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
		//TODO: substitute Gather with an All-to-All
		//TODO: or combine broadcasts in a single one -> DONE
		*/
		MPI_Bcast (&TlastK[0][0], Tcols, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
	}

	// last level (l=0)
	for (i=0; i<myTcols/2; i++)
	{
		for(rhs=0;rhs<m;rhs++)
		{
			xx[global[i]][rhs]=xx[global[i]][rhs]+Tlocal[0][i]*bb[0][rhs];
		}
	}

	// collect solution
	// MPI_IN_PLACE required for MPICH based versions
	if (rank==0)
	{
		MPI_Gather (MPI_IN_PLACE, 1, xx_rows_interleaved_resized, &xx[0][0], 1, xx_rows_interleaved_resized, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Gather (&xx[rank][0], 1, xx_rows_interleaved_resized, &xx[0][0], 1, xx_rows_interleaved_resized, 0, MPI_COMM_WORLD);
	}

	// cleanup
	free(local);
	free(global);
	free(map);

	DeallocateMatrix2D(TlastK,2,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Tlocal,n,CONTIGUOUS);
}
