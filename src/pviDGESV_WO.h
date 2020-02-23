#include <mpi.h>
#include <time.h>
#include "helpers/info.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "DGEZR.h"
//#include "pDGEIT_WX_async.h"
#include "pDGEIT_WX.h"

/*
 *	solve (SV) system with general (GE) matrix A of doubles (D)
 *	of order n and with m r.h.s in matrix bb[n,m] and solutions in xx[n,m]
 *	with:
 *	wide overwrite (WO) memory model
 *	parallelized in interleaved columns (pvi) over cprocs calculating processors
 *	parallelized initialization
 *	optimized loops
 *	ifs removed
 *
 */
result_info pviDGESV_WO(int n, double** A, int m, double** bb, double** xx, MPI_Comm comm)
{
	result_info wall_clock;

	wall_clock.total_start_time = time(NULL);

    int rank, cprocs; //
    MPI_Comm_rank(comm, &rank);		//get current process id
    MPI_Comm_size(comm, &cprocs);	// get number of processes

	int i,j,l;						// indexes
    int Tcols=2*n;					// total num of cols (X + K)
    int myTcols;					// num of T cols per process
    	myTcols=Tcols/cprocs;
    //int myAchunks;					// num of A rows/cols per process
    //	myAchunks=n/cprocs;
    int myxxrows=n/cprocs;
    int myKcols;
    	myKcols=myTcols/2;
    int myend;						// loop boundaries on local cols =myTcols/2;
    int mystart;
    int rhs;

    int avoidif;					// for boolean --> int conversion

    //int myAcols=n/cprocs;
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

	MPI_Status  mpi_status;
	MPI_Request mpi_request = MPI_REQUEST_NULL;


    /*
     * MPI derived types
     */

	/*
	 * option 1: derived data type, for interleaving while for gathering
	 *           non working for some values of n and np, why? extent of derived MPI data type?
	 */

	MPI_Datatype TlastKr_chunks;
	MPI_Type_vector (n/cprocs, 1, cprocs, MPI_DOUBLE, & TlastKr_chunks );
	MPI_Type_commit (& TlastKr_chunks);

	MPI_Datatype TlastKr_chunks_resized;
	MPI_Type_create_resized (TlastKr_chunks, 0, 1*sizeof(double), & TlastKr_chunks_resized);
	MPI_Type_commit (& TlastKr_chunks_resized);


	/*
	 * option 2: standard data type, explicit loop for interleaving after gathering
	 *           see below
	 */

	// rows of xx to be extracted
	MPI_Datatype xx_rows_interleaved;
	MPI_Type_vector (myxxrows, m, m*cprocs, MPI_DOUBLE, & xx_rows_interleaved );
	MPI_Type_commit (& xx_rows_interleaved);

	// rows of xx to be extracted, properly resized for gathering
	MPI_Datatype xx_rows_interleaved_resized;
	MPI_Type_create_resized (xx_rows_interleaved, 0, m*sizeof(double), & xx_rows_interleaved_resized);
	MPI_Type_commit (& xx_rows_interleaved_resized);


    /*
	 *  init inhibition table
	 */
	DGEZR(xx, n, m);															// init (zero) solution vectors
	pDGEIT_WX(A, Tlocal, TlastK, n, comm, rank, cprocs, map, global, local);	// init inhibition table
	//pDGEIT_W_async(A, Tlocal, TlastK, n, comm, rank, cprocs, map, global, local);// init inhibition table
    //MPI_Ibcast (&TlastK[0][0], n, MPI_DOUBLE, map[n-1], comm, &mpi_request);	// broadcast of the last col of T (K part)
    MPI_Bcast (&bb[0][0], n*m, MPI_DOUBLE, 0, comm);							// send all r.h.s to all procs

	/*
	 *  calc inhibition sequence
	 */
	wall_clock.core_start_time = time(NULL);

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		// ALL procs
		// update solutions
		// l .. n-1
		avoidif=(rank<map[l]);
		mystart = local[l] + avoidif;
		for (i=mystart; i<=local[n-1]; i++)
		{
			for (rhs=0;rhs<m;rhs++)
			{
				xx[global[i]][rhs]=xx[global[i]][rhs]+Tlocal[l][i]*bb[l][rhs];
			}
		}

		MPI_Wait(&mpi_request, &mpi_status);

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




		//////////////// update T
		// to avoid IFs: each process loops on its own set of cols, with indirect addressing

		//////// K
		// n .. n+l-1
		// ALL procs
		mystart=local[n];
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
		MPI_Igather (&Tlocal[l-1][local[n]], myKcols, MPI_DOUBLE, &TlastKr[0], 1, TlastKr_chunks_resized, map[l-1], comm, &mpi_request);
		// option 2: use last column buffer for temporary copy of non-interleaved data
		//MPI_Gather (&Tlocal[l-1][myKcols], myKcols, MPI_DOUBLE, &TlastKc[0], myKcols, MPI_DOUBLE, map[l-1], comm);

		//////// X
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

		// l .. n-1
		// ALL procs
		avoidif=(rank<map[l]);
		mystart=local[l]+avoidif;
		myend=local[n-1];
		for (i=0; i<=l-1; i++)
		{
			for (j=mystart; j<=myend; j++)
			{
				Tlocal[i][j]=Tlocal[i][j]*h[i] - Tlocal[l][j]*hh[i];
			}
		}

		//future last node broadcasts last row and col of K
		if (rank==map[l-1])
		{
			// copy data into local buffer before broadcast

			// option 1

			for (i=0; i<l-1; i++)
			{
				TlastKc[i]=Tlocal[i][local[n+l-1]];
			}


			// option 2
			/*
			myend=local[n+l-1];
			for (i=0; i<myKcols; i++)
			{
				ii=i*cprocs;
				for (j=0; j<cprocs; j++)
				{
					jj=j*myKcols+i;
					TlastKr[ii+j]=TlastKc[jj];		// interleave columns of the last row
					TlastKc[jj]=Tlocal[jj][myend];	// copy last column
				}
			}
			*/
		}
		// wait until gather completed
		MPI_Wait(&mpi_request, &mpi_status);
		//TODO: substitute Gather with an All-to-All
		MPI_Ibcast (&TlastK[0][0], Tcols, MPI_DOUBLE, map[l-1], comm, &mpi_request);
	}

	// last level (l=0)
	for (i=0; i<myxxrows; i++)
	{
		for(rhs=0;rhs<m;rhs++)
		{
			xx[global[i]][rhs]=xx[global[i]][rhs]+Tlocal[0][i]*bb[0][rhs];
		}
	}

	MPI_Wait(&mpi_request, &mpi_status);

    wall_clock.core_end_time = time(NULL);

	// collect solution
	// MPI_IN_PLACE required for MPICH based versions
	if (rank==0)
	{
		MPI_Gather (MPI_IN_PLACE, 1, xx_rows_interleaved_resized, &xx[0][0], 1, xx_rows_interleaved_resized, 0, comm);
	}
	else
	{
		MPI_Gather (&xx[rank][0], 1, xx_rows_interleaved_resized, &xx[0][0], 1, xx_rows_interleaved_resized, 0, comm);
	}

	// cleanup
	free(local);
	free(global);
	free(map);

	DeallocateMatrix2D(TlastK,2,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Tlocal,n,CONTIGUOUS);

	wall_clock.total_end_time = time(NULL);

	return wall_clock;
}
