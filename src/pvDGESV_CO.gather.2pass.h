#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "helpers/macros.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "DGEZR.h"
#include "pvDGEIT_CX.h"

/*
 *	solve (SV) system with general (GE) matrix A of doubles (D)
 *	of order n and with m r.h.s in matrix bb[n,m] and solutions in xx[n,m]
 *	with:
 *	compact overwrite (CO) memory model
 *	parallelized in NON-interleaved columns (pv) over cprocs calculating processors
 *	parallelized initialization
 *	non-optimized loops
 *	some overlapping calc/comm
 *
 */
test_output pvDGESV_CO_g_2pass(int nb, int n, double** A, int m, double** bb, double** xx, MPI_Comm comm)
{
	/*
	 * nb	blocking factor: number of adjacent column (block width)
	 * n	size (number of columns) of the square matrix A
	 * m	number of rigth-hand-sides (number of columns) in bb
	 *
	 */

	test_output result;

	result.total_start_time = time(NULL);

    int rank, cprocs; //
    MPI_Comm_rank(comm, &rank);		// get current process id
    MPI_Comm_size(comm, &cprocs);	// get number of processes
	MPI_Status  mpi_status;
	MPI_Request mpi_request = MPI_REQUEST_NULL;

	int i,j,l;						// general indexes
    int mycols   = n/cprocs;;		// num of cols per process
    int myxxrows = mycols;			// num of chunks for better code readability
    int myKcols  = mycols;
	int myXcols  = mycols;

    int rhs;

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Tlocal;
			 Tlocal=AllocateMatrix2D(n, mycols, CONTIGUOUS);
	// aliases
    // X part
    double** Xlocal;
    		 Xlocal=Tlocal;
    // K part
	double** Klocal;
			 Klocal=Tlocal;

	// last rows and cols of K
	double** lastK;
			 lastK=AllocateMatrix2D(2*nb, n, CONTIGUOUS);	// last rows [0 - (bf-1)] and cols [ bf - (2bf -1)] of K
	double** lastKr;
				lastKr=malloc(nb*sizeof(double*));
				for(i=0;i<nb;i++)
				{
					lastKr[i]=lastK[i];						// alias for last row
				}
	double** lastKc;
				lastKc=malloc(nb*sizeof(double*));
				for(i=0;i<nb;i++)
				{
					lastKc[i]=lastK[nb+i];					// alias for last col
				}
	// helper vectors
    double* h;
    		h=AllocateVector(n);
    double* hh;
			hh=AllocateVector(n);

    /*
     * MPI derived types
     */
	// interleaved nb chunks of a row of K, repeated for nb rows (that is: interleaved blocks of size (nb)x(nb) )
	MPI_Datatype lastKr_chunk;
	MPI_Type_vector (nb, myKcols, n, MPI_DOUBLE, & lastKr_chunk );
	MPI_Type_commit (& lastKr_chunk);

	// proper resizing for gathering
	MPI_Datatype lastKr_chunk_resized;
	MPI_Type_create_resized (lastKr_chunk, 0, myKcols*sizeof(double), & lastKr_chunk_resized);
	MPI_Type_commit (& lastKr_chunk_resized);

	// rows of xx to be extracted
	MPI_Datatype xx_chunk;
	MPI_Type_vector (myxxrows/nb, m*nb, nb*m*cprocs, MPI_DOUBLE, & xx_chunk );
	MPI_Type_commit (& xx_chunk);

	// rows of xx to be extracted, properly resized for gathering
	MPI_Datatype xx_chunk_resized;
	MPI_Type_create_resized (xx_chunk, 0, nb*m*sizeof(double), & xx_chunk_resized);
	MPI_Type_commit (& xx_chunk_resized);


    /*
	 *  init inhibition table
	 */
	DGEZR(xx, n, m);												// init (zero) solution vectors
	pvDGEIT_CX(A, Tlocal, lastK, n, nb, comm, rank, cprocs);	// init inhibition table

	/*
	MPI_Barrier(MPI_COMM_WORLD);
	for (i=0; i<cprocs; i++)
	{
		if (rank==i)
		{
			printf("%d-%d:\n",l,rank);
			PrintMatrix2D(Tlocal, n, mycols);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	*/

	MPI_Bcast (&bb[0][0], n*m, MPI_DOUBLE, 0, comm);								// send all r.h.s to all procs

	/*
	 *  calc inhibition sequence
	 */
	result.core_start_time = time(NULL);

	// general bounds for the loops over the columns
	// they differ on processes and change along the main loop over the levels
	int myKend = myKcols-1;		// position of the last col of K
	int myXmid = myXcols-1; 	// position of boundary between the left (simplified topological formula) and right (full formula) part of X
	int myxxstart = myXcols;	// beginning column position for updating the solution (begins from right)
	int current_last=nb-1;		// index for the current last row or col of K in buffer

	int gi;
	int talker;

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{

		/*
		MPI_Barrier(MPI_COMM_WORLD);
		for (i=0; i<1; i++)
		{
			if (rank==1)
			{
				printf("%d-%d:\n",l,rank);
				PrintMatrix2D(Tlocal, n, mycols);
				fflush(stdout);
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
		*/

		if (rank==PVMAP(l, mycols)) // if a process contains the last rows/cols, must skip it
		{
			myKend--;
			myXmid--;
			myxxstart--;
		}

		// update solutions
		// l .. n-1
		for (i=myxxstart; i<=PVLOCAL(n-1, mycols); i++)
		{
			gi=PVGLOBAL(i, mycols, rank);
			for (rhs=0;rhs<m;rhs++)
			{
				// on column < l X is null and not stored in T
				//if (global[i]>=l) xx[global[i]][rhs]=xx[global[i]][rhs]+Xlocal[l][i]*bb[l][rhs];
				xx[gi][rhs]=xx[gi][rhs]+Xlocal[l][i]*bb[l][rhs];
			}
		}

		// wait for new last rows and cols before computing helpers
		if (current_last==nb-1) MPI_Wait(&mpi_request, &mpi_status);
		// TODO: check performance penalty by skipping MPI_wait with an 'if' for non due cases (inside the blocking factor) or not

		// update helpers
		for (i=0; i<=l-1; i++)
		{
			h[i]   = 1/(1-lastKc[current_last][i]*lastKr[current_last][i]);
			hh[i]  = lastKc[current_last][i]*h[i];
			for (rhs=0;rhs<m;rhs++)
			{
				bb[i][rhs] = bb[i][rhs]-lastKr[current_last][i]*bb[l][rhs];
			}
		}

		//////// K
		// 0 .. l-1
		for (i=0; i<=l-1; i++)
		{
			for (j=0; j<=myKend; j++)
			{
				// on the diagonal T stores X values
				gi=PVGLOBAL(j, mycols, rank);
				if (gi!=i) Klocal[i][j]=Klocal[i][j]*h[i] - Klocal[l][j]*hh[i];
			}
		}


		///////// update local copy of global last rows and cols of K
		if (current_last>0) // block of last rows (cols) not completely scanned
		{
			for (i=0;i<current_last;i++)
			{
				for (j=0; j<=l-1; j++)
				{
					lastKc[i][j]=lastKc[i][j]*h[j]   - lastKr[current_last][l-current_last+i]*hh[j];
					lastKr[i][j]=lastKr[i][j]*h[l-current_last+i] - lastKr[current_last][j]*hh[l-current_last+i];
				}
			}
			current_last--; // update counter to point to the "future" last row (col) in the block
		}
		else // block of last rows (cols) completely scanned
		{
			current_last=nb-1; // reset counter for next block (to be sent/received)
			{
				talker = PVMAP(l-nb, myKcols);
				// collect chunks of last row of K to "future" last node
				MPI_Igather (&Klocal[l-nb][PVLOCAL(0, myKcols)], nb*myKcols, MPI_DOUBLE, &lastKr[0][0], 1, lastKr_chunk_resized, talker, comm, &mpi_request);

				//future last node broadcasts last rows and cols of K
				if (rank==talker)
				{
					// copy data into local buffer before broadcast
					for(j=0;j<nb;j++)
					{
						for (i=0; i<=l-1; i++)
						{
							lastKc[j][i]=Klocal[i][PVLOCAL(l-nb, myKcols)+j];
						}
					}
					// wait for gathering to complete
					MPI_Wait(&mpi_request, &mpi_status);
				}
				// do not wait all for gather: only who has to broadcast
				//MPI_Wait(&mpi_request, &mpi_status);
				MPI_Ibcast (&lastK[0][0], 2*n*nb, MPI_DOUBLE, talker, comm, &mpi_request);
			}
		}

		//////// X
		//// 0 .. l-1
		// calc with diagonal elements not null (left part of X)
		for (i=0; i<=myXmid; i++)
		{
			gi=PVGLOBAL(i, mycols, rank);
			Xlocal[gi][i]=Xlocal[gi][i]*h[gi];
		}

		// l .. n-1
		// calc with general elements (right part of X)
		// must differentiate topological formula on special column l
		if (rank==PVMAP(l, myKcols))
		{
			for (i=0; i<=l-1; i++)
			{
				j=myXmid+1;
				{
					Xlocal[i][j]= - Xlocal[l][j]*hh[i];
				}
			}
			for (i=0; i<=l-1; i++)
			{
				for (j=myXmid+2; j<=myXcols-1; j++)
				{
					Xlocal[i][j]=Xlocal[i][j]*h[i] - Xlocal[l][j]*hh[i];
				}
			}
		}
		else
		{
			for (i=0; i<=l-1; i++)
			{
				for (j=myXmid+1; j<=myXcols-1; j++)
				{
					Xlocal[i][j]=Xlocal[i][j]*h[i] - Xlocal[l][j]*hh[i];
				}
			}
		}
	}

	/*
	MPI_Barrier(MPI_COMM_WORLD);
	for (i=0; i<cprocs; i++)
	{
		if (rank==i)
		{
			printf("%d-%d:\n",l,rank);
			PrintMatrix2D(Tlocal, n, mycols);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	*/

	// last level (l=0)
	for (i=0; i<myxxrows; i++)
	{
		gi=PVGLOBAL(i, mycols, rank);
		for(rhs=0;rhs<m;rhs++)
		{
			xx[gi][rhs]=xx[gi][rhs]+Xlocal[0][i]*bb[0][rhs];
		}
	}

	//MPI_Wait(&mpi_request, &mpi_status);

	result.core_end_time = time(NULL);

	// TODO: add checking on exit code
	result.exit_code = 0;

	// collect solution
	// MPI_IN_PLACE required for MPICH based versions
	if (rank==0)
	{
		MPI_Gather (MPI_IN_PLACE, m*myxxrows, MPI_DOUBLE, &xx[0][0], m*myxxrows, MPI_DOUBLE, 0, comm);
	}
	else
	{
		MPI_Gather (&xx[rank*myxxrows][0], m*myxxrows, MPI_DOUBLE, &xx[0][0], m*myxxrows, MPI_DOUBLE, 0, comm);
	}

	/*
	 * control on factorization
	 */
	/*
	MPI_Datatype Tlocal_half;
	MPI_Type_vector (n*myKcols, 1, 1, MPI_DOUBLE, & Tlocal_half );
	MPI_Type_commit (& Tlocal_half);

	MPI_Datatype Thalf_interleaved;
	MPI_Type_vector (n*myKcols/nb, nb, nb*cprocs, MPI_DOUBLE, & Thalf_interleaved );
	MPI_Type_commit (& Thalf_interleaved);

	MPI_Datatype Thalf_interleaved_resized;
	MPI_Type_create_resized (Thalf_interleaved, 0, nb*sizeof(double), & Thalf_interleaved_resized);
	MPI_Type_commit (& Thalf_interleaved_resized);

	MPI_Gather (&Xlocal[0][0], 1, Tlocal_half, &A[0][0], 1, Thalf_interleaved_resized, 0, comm);

	MPI_Barrier(comm);
	if (rank==0 )
	{
		printf("\n\n Matrix X:\n");
		PrintMatrix2D(A, n, n);
		fflush(stdout);
	}
	*/

	MPI_Wait(&mpi_request, &mpi_status);
	MPI_Barrier(MPI_COMM_WORLD);

	// cleanup
	NULLFREE(lastKc);
	NULLFREE(lastKr);

	DeallocateMatrix2D(lastK,2*nb,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Tlocal,n,CONTIGUOUS);

	result.total_end_time = time(NULL);

	return result;
}
