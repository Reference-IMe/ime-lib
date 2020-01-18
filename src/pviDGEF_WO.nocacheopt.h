#include <mpi.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "pDGEIT_WX.h"

/*
 *	factorization (F) of a general (GE) matrix A of doubles (D)
 *	of order n
 *	with:
 *	wide overwrite (WO) memory model
 *	parallelized in interleaved columns (pvi) over cprocs calculating processors
 *	parallelized initialization
 *	optimized loops
 *	ifs removed
 *
 */
void pviDGEF_WO(int n, double** A, double** K, MPI_Comm comm)
{
    int rank, cprocs; //
    MPI_Comm_rank(comm, &rank);		//get current process id
    MPI_Comm_size(comm, &cprocs);	// get number of processes

	int i,j,l;						// indexes
    int Tcols=2*n;					// total num of cols (X + K)
    int myTcols;					// num of T cols per process
    	myTcols=Tcols/cprocs;
    int myKcols;					// num of K (or X) cols per process
    	myKcols=myTcols/2;
    int myend;						// loop boundaries on local cols =myTcols/2;
    int mystart;

    int avoidif;					// for boolean --> int conversion

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
	MPI_Datatype Tlocal_half;
	MPI_Type_vector (n, myKcols, myTcols, MPI_DOUBLE, & Tlocal_half );
	MPI_Type_commit (& Tlocal_half);

	MPI_Datatype Thalf_interleaved;
	MPI_Type_vector (n*myKcols, 1, cprocs, MPI_DOUBLE, & Thalf_interleaved );
	MPI_Type_commit (& Thalf_interleaved);

	MPI_Datatype Thalf_interleaved_resized;
	MPI_Type_create_resized (Thalf_interleaved, 0, 1*sizeof(double), & Thalf_interleaved_resized);
	MPI_Type_commit (& Thalf_interleaved_resized);

    /*
	 *  init inhibition table
	 */
	pDGEIT_W(A, Tlocal, TlastK, n, comm, rank, cprocs, map, global, local);	// init inhibition table

	/*
	 *  calc inhibition sequence
	 */

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		// ALL procs
		// update solutions
		// l .. n-1

		//avoidif=(rank<map[l]);
		//mystart = local[l] + avoidif;

		// ALL procs
		// update helpers
		for (i=0; i<=l-1; i++)
		{
			h[i]   = 1/(1-TlastKc[i]*TlastKr[i]);
			hh[i]  = TlastKc[i]*h[i];
		}

		// update T
		// to avoid IFs: each process loops on its own set of cols, with indirect addressing

		// 0 .. l-1
		// ALL procs
		// processes with diagonal elements not null
		mystart=0;
		avoidif = (rank>map[l-1]);
		myend = local[l-1] - avoidif;
		for (i=mystart; i<=myend; i++)
		{
			Tlocal[global[i]][i]=Tlocal[global[i]][i]*h[global[i]];
		}

		// l .. n+l-1
		// ALL procs
		avoidif=(rank<map[l]);
		mystart=local[l]+avoidif;
		avoidif=(rank>map[n+l-1]);
		myend=local[n+l-1]-avoidif;
		for (i=0; i<=l-1; i++)
		{
			for (j=mystart; j<=myend; j++)
			{
				Tlocal[i][j]=Tlocal[i][j]*h[i] - Tlocal[l][j]*hh[i];
			}
		}

		// collect chunks of last row of K to "future" last node

		// option 1: non working
		//MPI_Gather (&Tlocal[l-1][local[n]], myTcols/2, MPI_DOUBLE, &TlastKr[0], 1, TlastKr_chunks_resized, map[l-1], comm);
		// option 2: use last column buffer for temporary copy of non-interleaved data
		MPI_Gather (&Tlocal[l-1][myKcols], myKcols, MPI_DOUBLE, &TlastKc[0], myKcols, MPI_DOUBLE, map[l-1], comm);

		//future last node broadcasts last row and col of K
		if (rank==map[l-1])
		{
			// copy data into local buffer before broadcast

			// option 1
			/*
			for (i=0; i<n; i++)
			{
				TlastKc[i]=Tlocal[i][local[n+l-1]];
			}
			*/

			// option 2
			myend=local[n+l-1];
			for (i=0; i<myKcols; i++)
			{
				int ii=i*cprocs;
				for (j=0; j<cprocs; j++)
				{
					int jj=j*myKcols+i;
					TlastKr[ii+j]=TlastKc[jj];		// interleave columns of the last row
					TlastKc[jj]=Tlocal[jj][myend];	// copy last column
				}
			}
		}

		//TODO: substitute Gather with an All-to-All
		MPI_Bcast (&TlastK[0][0], Tcols, MPI_DOUBLE, map[l-1], comm);
	}


	MPI_Gather (&Tlocal[0][0], 1, Tlocal_half, &A[0][0], 1, Thalf_interleaved_resized, 0, comm);
	MPI_Gather (&Tlocal[0][myKcols], 1, Tlocal_half, &K[0][0], 1, Thalf_interleaved_resized, 0, comm);


	// cleanup
	free(local);
	free(global);
	free(map);

	DeallocateMatrix2D(TlastK,2,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	//DeallocateMatrix2D(Tlocal,n,CONTIGUOUS);
}
