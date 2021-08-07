#include <mpi.h>
#include <time.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "DGEZR.h"
#include "pvDGEIT_WX_non1D.h"

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
test_output pviDGESV_WO(int n, double** A, int m, double** bb, double** xx, MPI_Comm comm)
{
	test_output result;

	result.total_start_time = time(NULL);

    int rank, cprocs; //
    MPI_Comm_rank(comm, &rank);		//get current process id
    MPI_Comm_size(comm, &cprocs);	// get number of processes

	int i,j,l;						// indexes
    int mycols;					// num of T cols per process
    	mycols=2*n/cprocs;
    int myxxrows=n/cprocs;
    int myKcols;
    	myKcols=mycols/2;
	int myXcols;
		myXcols=mycols/2;
    int myend;						// loop boundaries on local cols =myTcols/2;
    int mystart;
    int rhs;

    int avoidif;					// for boolean --> int conversion

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Xlocal;
    		 Xlocal=AllocateMatrix2D(n, myXcols, CONTIGUOUS);
	double** Klocal;
			 Klocal=AllocateMatrix2D(n, myKcols, CONTIGUOUS);
	double** lastK;
			 lastK=AllocateMatrix2D(2,n, CONTIGUOUS);	// last col [0] and row [1] of T (K part)
	double*  lastKc=&lastK[0][0];						// alias for last col
	double*  lastKr=&lastK[1][0];						// alias for last row

    double* h;							// helper vectors
    		h=AllocateVector(n);
    double* hh;
			hh=AllocateVector(n);

	/*
	 * map columns to process
	 */
	int*	local;
    		local=malloc(n*sizeof(int));
    int*	map;
    		map=malloc(n*sizeof(int));
			for (i=0; i<n; i++)
			{
				map[i]= i % cprocs;			// who has the col i
				local[i]=floor(i/cprocs);	// position of the column i(global) in the local matrix
			}
    int*	global;
    		global=malloc(mycols*sizeof(int));
			for (i=0; i<mycols; i++)
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

	MPI_Datatype lastKr_chunks;
	MPI_Type_vector (myKcols, 1, cprocs, MPI_DOUBLE, & lastKr_chunks );
	MPI_Type_commit (& lastKr_chunks);

	MPI_Datatype lastKr_chunks_resized;
	MPI_Type_create_resized (lastKr_chunks, 0, 1*sizeof(double), & lastKr_chunks_resized);
	MPI_Type_commit (& lastKr_chunks_resized);


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
	pvDGEIT_WX(A, Xlocal, Klocal, lastK, n, comm, rank, cprocs, map, global, local);	// init inhibition table
    MPI_Bcast (&bb[0][0], n*m, MPI_DOUBLE, 0, comm);							// send all r.h.s to all procs


	/*
	 *  calc inhibition sequence
	 */
	result.core_start_time = time(NULL);

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
				xx[global[i]][rhs]=xx[global[i]][rhs]+Xlocal[l][i]*bb[l][rhs];
			}
		}

		MPI_Wait(&mpi_request, &mpi_status);

		// ALL procs
		// update helpers
		for (i=0; i<=l-1; i++)
		{
			h[i]   = 1/(1-lastKc[i]*lastKr[i]);
			hh[i]  = lastKc[i]*h[i];
			for (rhs=0;rhs<m;rhs++)
			{
				bb[i][rhs] = bb[i][rhs]-lastKr[i]*bb[l][rhs];
			}
		}


		//////////////// update T
		// to avoid IFs: each process loops on its own set of cols, with indirect addressing

		//////// K
		// 0 .. l-1
		// ALL procs
		mystart=local[0];
		avoidif=(rank>map[l-1]);
		myend=local[l-1]-avoidif;
		for (i=0; i<=l-1; i++)
		{
			for (j=mystart; j<=myend; j++)
			{
				Klocal[i][j]=Klocal[i][j]*h[i] - Klocal[l][j]*hh[i];
			}
		}

		// collect chunks of last row of K to "future" last node
		MPI_Igather (&Klocal[l-1][local[0]], myKcols, MPI_DOUBLE, &lastKr[0], 1, lastKr_chunks_resized, map[l-1], comm, &mpi_request);

		//////// X
		// 0 .. l-1
		// ALL procs
		// processes with diagonal elements not null
		mystart=0;
		avoidif = (rank>map[l-1]);
		myend = local[l-1] - avoidif;
		for (i=mystart; i<=myend; i++)
		{
			Xlocal[global[i]][i]=Xlocal[global[i]][i]*h[global[i]];
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
				Xlocal[i][j]=Xlocal[i][j]*h[i] - Xlocal[l][j]*hh[i];
			}
		}

		//future last node broadcasts last row and col of K
		if (rank==map[l-1])
		{
			// copy data into local buffer before broadcast

			// option 1

			for (i=0; i<l-1; i++)
			{
				lastKc[i]=Klocal[i][local[l-1]];
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
		MPI_Ibcast (&lastK[0][0], 2*n, MPI_DOUBLE, map[l-1], comm, &mpi_request);
	}

	// last level (l=0)
	for (i=0; i<myxxrows; i++)
	{
		for(rhs=0;rhs<m;rhs++)
		{
			xx[global[i]][rhs]=xx[global[i]][rhs]+Xlocal[0][i]*bb[0][rhs];
		}
	}

	MPI_Wait(&mpi_request, &mpi_status);

    result.core_end_time = time(NULL);
	result.exit_code = 0;

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

	DeallocateMatrix2D(lastK,2,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Xlocal,n,CONTIGUOUS);
	DeallocateMatrix2D(Klocal,n,CONTIGUOUS);

	result.total_end_time = time(NULL);

	return result;
}
