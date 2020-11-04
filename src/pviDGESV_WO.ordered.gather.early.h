#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "helpers/macros.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "DGEZR.h"
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
test_output pviDGESV_WO_oge(int nb, int n, double** A, int m, double** bb, double** xx, MPI_Comm comm)
{
	/*
	 * nb	blocking factor: number of adjacent column (block width)
	 * n	size (number of columns) of the square matrix A
	 * m	number of rigth-hand-sides (number of columns) in bb
	 *
	 */

	test_output result;

	result.total_start_time = time(NULL);

    int rank, cprocs;
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
    // X part
    double** Xlocal;
    		 Xlocal=AllocateMatrix2D(n, myXcols, CONTIGUOUS);
    // K part
	double** Klocal;
			 Klocal=AllocateMatrix2D(n, myKcols, CONTIGUOUS);
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
	// helper vectors (matrices indeed, one row for each level inside the block (of the blocking factor)
	double** h_block;
			 h_block=AllocateMatrix2D(nb,n,CONTIGUOUS);
	double** hh_block;
			 hh_block=AllocateMatrix2D(nb,n,CONTIGUOUS);
	/*
	 * map columns to process
	 */
	int*	local;
    		local=malloc(n*sizeof(int));
    int*	map;
    		map=malloc(n*sizeof(int));
			for (i=0; i<n; i++)
			{
				map[i]= ((int)floor(i/nb)) % cprocs;			// who has the col i
				local[i]=(int)floor(i/(nb*cprocs))*nb + i % nb;	// position of the column i(global) in the local matrix
			}
    int*	global;
    		global=malloc(mycols*sizeof(int));
			for (i=0; i<mycols; i++)
			{
				global[i]= i % nb + (int)floor(i/nb)*cprocs*nb + rank*nb; // position of the column i(local) in the global matrix
			}

    /*
     * MPI derived types
     */
	// interleaved nb chunks of a row of K, repeated for nb rows (that is: interleaved blocks of size (nb)x(nb) )
	MPI_Datatype multiple_lastKr_chunks;
	MPI_Type_vector (nb*(myKcols/nb), nb, nb*cprocs, MPI_DOUBLE, & multiple_lastKr_chunks );
	MPI_Type_commit (& multiple_lastKr_chunks);

	// proper resizing for gathering
	MPI_Datatype multiple_lastKr_chunks_resized;
	MPI_Type_create_resized (multiple_lastKr_chunks, 0, nb*sizeof(double), & multiple_lastKr_chunks_resized);
	MPI_Type_commit (& multiple_lastKr_chunks_resized);

	// rows of xx to be extracted
	MPI_Datatype xx_rows_interleaved;
	MPI_Type_vector (myxxrows/nb, m*nb, nb*m*cprocs, MPI_DOUBLE, & xx_rows_interleaved );
	MPI_Type_commit (& xx_rows_interleaved);

	// rows of xx to be extracted, properly resized for gathering
	MPI_Datatype xx_rows_interleaved_resized;
	MPI_Type_create_resized (xx_rows_interleaved, 0, nb*m*sizeof(double), & xx_rows_interleaved_resized);
	MPI_Type_commit (& xx_rows_interleaved_resized);


    /*
	 *  init inhibition table
	 */
	DGEZR(xx, n, m);																		// init (zero) solution vectors
	pDGEIT_WX(A, Xlocal, Klocal, lastK, n, nb, comm, rank, cprocs, map, global, local);		// init inhibition table
    MPI_Bcast (&bb[0][0], n*m, MPI_DOUBLE, 0, comm);										// send all r.h.s to all procs

	/*
	 *  calc inhibition sequence
	 */
	result.core_start_time = time(NULL);

	// general bounds for the loops over the columns
	// they differ on processes and change along the main loop over the levels
	int myKend = myKcols-1;		// position of the last col of K
	int myXmid = myXcols-1;		// position of boundary between the left (simplified topological formula) and right (full formula) part of X
	int myxxstart = myXcols;	// beginning column position for updating the solution (begins from right)
	int current_last=nb-1;		// index for the current last row or col of K in buffer

	int l_block;				// inhibition level inside the block (of the blocking factor)

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		if (rank==map[l]) // if a process contains the last rows/cols, must skip it
		{
			myKend--;
			myXmid--;
			myxxstart--;
		}

		// update solutions
		// l .. n-1
		// if myxxstart==myXcols, skip the loop
		for (i=myxxstart; i<=local[n-1]; i++)
		{
			for (rhs=0;rhs<m;rhs++)
			{
				xx[global[i]][rhs]=xx[global[i]][rhs]+Xlocal[l][i]*bb[l][rhs];
			}
		}

		// wait for new last rows and cols before computing helpers
		//MPI_Wait(&mpi_request, &mpi_status);

		if (current_last==nb-1)
		{
			MPI_Wait(&mpi_request, &mpi_status);

			l_block=l;
			int cl;
			for (cl=current_last; cl>=0; cl--) // for every row in the block
			{
				// update helpers
				for (i=0; i<=l_block-1; i++)
				{
					h_block[cl][i]   = 1/(1-lastKc[cl][i]*lastKr[cl][i]);
					hh_block[cl][i]  = lastKc[cl][i]*h_block[cl][i];
				}
				// update auxiliary quantities
				for (i=0; i<=l_block-1; i++)
				{
					for (rhs=0;rhs<m;rhs++)
					{
						bb[i][rhs] = bb[i][rhs]-lastKr[cl][i]*bb[l_block][rhs];
					}
				}
				for (i=0;i<cl;i++) // not executed for row 0
				{
					for (j=0; j<=l_block-1; j++)
					{
						lastKc[i][j]=lastKc[i][j]*h_block[cl][j]                      - lastKr[cl][l_block-cl+i]*hh_block[cl][j];
						lastKr[i][j]=lastKr[i][j]*h_block[cl][l_block-cl+i] - lastKr[cl][j]                     *hh_block[cl][l_block-cl+i];
					}
				}
				l_block--;
			}
		}

		//////// K
		//// 0 .. l-1 (reversed)
		// separating last nb rows (to be sent) from the remainder
		if ( (l-nb) >= 0 ) // it's not the last block (=> gathering required)
		{
			// local last nb rows of K
			for (i=l-1; i>=l-nb; i--)
			{
				for (j=0; j<=myKend; j++)
				{
					Klocal[i][j]=Klocal[i][j]*h_block[current_last][i] - Klocal[l][j]*hh_block[current_last][i];
				}
			}

			// early gathering of last nb rows of K
			if (current_last == 0) // if the block of nb rows has been completely scanned
			{
				MPI_Igather (&Klocal[l-nb][local[0]], nb*myKcols, MPI_DOUBLE, &lastKr[0][0], 1, multiple_lastKr_chunks_resized, map[l-nb], comm, &mpi_request);
			}

			// local remaining rows of K
			for (i=l-nb-1; i>=0; i--) // not executed if the end (beginning) of the block coincide with the end (beginning) of the matrix
			{
				for (j=0; j<=myKend; j++)
				{
					Klocal[i][j]=Klocal[i][j]*h_block[current_last][i] - Klocal[l][j]*hh_block[current_last][i];
				}
			}
		}
		else // it's the last block (=> gathering not required)
		{
			for (i=l-1; i>=0; i--)
			{
				for (j=0; j<=myKend; j++)
				{
					Klocal[i][j]=Klocal[i][j]*h_block[current_last][i] - Klocal[l][j]*hh_block[current_last][i];
				}
			}
		}

		if (current_last == 0) // if the block of nb rows has been completely scanned
		{
			//future last node broadcasts last rows and cols of K
			if (rank==map[l-nb])
			{
				// copy data into local buffer before broadcast
				for(j=0;j<nb;j++)
				{
					for (i=0; i<=l-1; i++)
					{
						lastKc[j][i]=Klocal[i][local[l-nb]+j];
					}
				}
				// wait for gathering to complete
				MPI_Wait(&mpi_request, &mpi_status);
			}
			// do not wait all for gather: only who has to broadcast
			//MPI_Wait(&mpi_request, &mpi_status);
			MPI_Ibcast (&lastK[0][0], 2*n*nb, MPI_DOUBLE, map[l-nb], comm, &mpi_request);

			//////// X
			//// 0 .. l-1
			// calc with diagonal elements not null (left part of X)
			for (i=0; i<=myXmid; i++)
			{
				Xlocal[global[i]][i]=Xlocal[global[i]][i]*h_block[current_last][global[i]];
			}
			//////// X
			//// l .. n-1
			// calc with general elements (right part of X)
			for (i=0; i<=l-1; i++)
			{
				for (j=myXmid+1; j<=myXcols-1; j++)
				{
					Xlocal[i][j]=Xlocal[i][j]*h_block[current_last][i] - Xlocal[l][j]*hh_block[current_last][i];
				}
			}

			current_last=nb-1;
		}
		else // block of last rows (cols) not completely scanned
		{
			//////// X
			//// 0 .. l-1
			// calc with diagonal elements not null (left part of X)
			for (i=0; i<=myXmid; i++)
			{
				Xlocal[global[i]][i]=Xlocal[global[i]][i]*h_block[current_last][global[i]];
			}
			//////// X
			//// l .. n-1
			// calc with general elements (right part of X)
			for (i=0; i<=l-1; i++)
			{
				for (j=myXmid+1; j<=myXcols-1; j++)
				{
					Xlocal[i][j]=Xlocal[i][j]*h_block[current_last][i] - Xlocal[l][j]*hh_block[current_last][i];
				}
			}

			current_last--; // update counter to point to the "future" last row (col) in the block
		}
	}

	// last level (l=0)
	for (i=0; i<myxxrows; i++)
	{
		for(rhs=0;rhs<m;rhs++)
		{
			xx[global[i]][rhs]=xx[global[i]][rhs]+Xlocal[0][i]*bb[0][rhs];
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
		MPI_Gather (MPI_IN_PLACE, 1, xx_rows_interleaved_resized, &xx[0][0], 1, xx_rows_interleaved_resized, 0, comm);
	}
	else
	{
		MPI_Gather (&xx[rank*nb][0], 1, xx_rows_interleaved_resized, &xx[0][0], 1, xx_rows_interleaved_resized, 0, comm);
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
	MPI_Barrier(comm);

	// cleanup
	NULLFREE(local);
	NULLFREE(global);
	NULLFREE(map);
	NULLFREE(lastKc);
	NULLFREE(lastKr);

	DeallocateMatrix2D(lastK,2*nb,CONTIGUOUS);
	DeallocateMatrix2D(h_block,nb,CONTIGUOUS);
	DeallocateMatrix2D(hh_block,nb,CONTIGUOUS);
	DeallocateMatrix2D(Xlocal,n,CONTIGUOUS);
	DeallocateMatrix2D(Klocal,n,CONTIGUOUS);

	result.total_end_time = time(NULL);

	return result;
}
