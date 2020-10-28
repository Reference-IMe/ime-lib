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
 * initial tentative for an "early" version
 */
test_output pvDGESV_CO_dev(int nb, int n, double** A, int m, double** bb, double** xx, MPI_Comm comm)
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
	MPI_Status  mpi_status[2];
	MPI_Request mpi_request[2];
				mpi_request[0] = MPI_REQUEST_NULL; // req. for allgather
				mpi_request[1] = MPI_REQUEST_NULL; // req. for broadcast
	int i,j,l;						// general indexes
    int mycols   = n/cprocs;;		// num of cols per process
    int myxxrows = mycols;			// num of chunks for better code readability
    int rhs;

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Tlocal;
			 Tlocal=AllocateMatrix2D(n, mycols, CONTIGUOUS);
	// aliases for better code readability
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
	// helper vectors (matrices indeed, one row for each level inside the block (of the blocking factor)
	double** h_block;
			 h_block=AllocateMatrix2D(nb,n,CONTIGUOUS);
	double** hh_block;
			 hh_block=AllocateMatrix2D(nb,n,CONTIGUOUS);

    /*
     * MPI derived types
     */
	// last cols of K
	MPI_Datatype lastKc_col;
	MPI_Type_vector (nb, 1, n, MPI_DOUBLE, & lastKc_col );
	MPI_Type_commit (& lastKc_col);

	// last rows of K (different type for s-end and r-eceive)
	MPI_Datatype s_lastKr_chunk;
	MPI_Type_vector (nb, 1, mycols, MPI_DOUBLE, & s_lastKr_chunk );
	MPI_Type_commit (& s_lastKr_chunk);

	MPI_Datatype r_lastKr_chunk;
	MPI_Type_vector (nb, 1, n, MPI_DOUBLE, & r_lastKr_chunk );
	MPI_Type_commit (& r_lastKr_chunk);

	// proper resizing for gathering
	MPI_Datatype lastKc_col_resized;
	MPI_Type_create_resized (lastKc_col, 0, 1*sizeof(double), & lastKc_col_resized);
	MPI_Type_commit (& lastKc_col_resized);

	MPI_Datatype s_lastKr_chunk_resized;
	MPI_Type_create_resized (s_lastKr_chunk, 0, 1*sizeof(double), & s_lastKr_chunk_resized);
	MPI_Type_commit (& s_lastKr_chunk_resized);

	MPI_Datatype r_lastKr_chunk_resized;
	MPI_Type_create_resized (r_lastKr_chunk, 0, 1*sizeof(double), & r_lastKr_chunk_resized);
	MPI_Type_commit (& r_lastKr_chunk_resized);

	// rows of xx to be extracted
	MPI_Datatype xx_chunk;
	MPI_Type_vector (myxxrows/nb, m*nb, nb*m*cprocs, MPI_DOUBLE, & xx_chunk );
	MPI_Type_commit (& xx_chunk);

	// rows of xx to be extracted, properly resized for gathering
	MPI_Datatype xx_chunk_resized;
	MPI_Type_create_resized (xx_chunk, 0, nb*m*sizeof(double), & xx_chunk_resized);
	MPI_Type_commit (& xx_chunk_resized);

	int* gather_count;
		 gather_count=malloc(cprocs*sizeof(int));

	int* gather_displacement;
		 gather_displacement=malloc(cprocs*sizeof(int));

		 for (i=0; i<cprocs; i++)
		 {
			gather_displacement[i]=i*mycols;
			gather_count[i]=mycols;
		 }
    /*
	 *  init inhibition table
	 */
	DGEZR(xx, n, m);												// init (zero) solution vectors
	pvDGEIT_CX(A, Tlocal, lastK, n, nb, comm, rank, cprocs);	// init inhibition table

	// last proc has already sent a full chunk of lastKr
	// decrease chunk size by nb for next sending
	gather_count[cprocs-1]=mycols-nb;

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
	int myxxstart = mycols;		// beginning column position for updating the solution (begins from right)
	int current_last=nb-1;		// index for the current last row or col of K in buffer
	int firstdiag=rank*mycols;	// (global) position of the first diagonal element on this rank
	int l_col;					// (local) position of the column l
	int l_rank_future;			// rank holding the next (in future) block of last nb column
	int l_rank;					// rank holdinh the current l column
	int gi;						// global index
	int l_block;				// inhibition level inside the block (of the blocking factor)

	//TODO: pre-calc other values..

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		/*
		MPI_Barrier(MPI_COMM_WORLD);
		for (i=0; i<1; i++)
		{
			if (rank==0)
			{
				printf("%d-%d:\n",l,rank);
				PrintMatrix2D(Tlocal, n, mycols);
				fflush(stdout);
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
		*/

		l_col  = PVLOCAL(l, mycols);
		l_rank = PVMAP(l, mycols);

		if (rank==l_rank) // if a process contains the last rows/cols, must skip it
		{
			myxxstart--;
		}
		//TODO: avoid if?

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

		if (current_last==nb-1)
		{
			MPI_Waitall(2, mpi_request, mpi_status);

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

		// TODO: insert EARLY gathering of last cols/rows

		// must differentiate topological formula on special column l
		//
		if (rank==l_rank)			// proc. containing column l
		{//rows:
			// remaining
			for (i=l-1; i>=firstdiag+l_col; i--)
			{
				//printf("CC (%d) %d:%d\n",rank,l,i);

				// after column l (X values)
				for (j=mycols-1; i>l_col; j--)
				{
					Xlocal[i][j]=Xlocal[i][j]*h_block[current_last][i] - Xlocal[l][j]*hh_block[current_last][i];
				}

				// column l (X value)
				Xlocal[i][l_col]= - Xlocal[l][l_col]*hh_block[current_last][i];

				// before column l (K values)
				for (j=l_col-1; j>=0; j--)
				{
					Klocal[i][j]=Klocal[i][j]*h_block[current_last][i] - Klocal[l][j]*hh_block[current_last][i];
				}
			}
			// containing first diagonal element
			for (i=firstdiag+l_col-1; i>=firstdiag; i--)
			{//columns:
				//printf("BB (%d) %d:%d\n",rank,l,i);

				// after column l (X values)
				for (j=mycols-1; j>l_col; j--)
				{
					Xlocal[i][j]=Xlocal[i][j]*h_block[current_last][i] - Xlocal[l][j]*hh_block[current_last][i];
				}

				// column l (X value)
				Xlocal[i][l_col]= - Xlocal[l][l_col]*hh_block[current_last][i];

				// after diagonal element and before column l (K values)
				for (j=l_col-1; j>i-firstdiag; j--)
				{
					Klocal[i][j]=Klocal[i][j]*h_block[current_last][i] - Klocal[l][j]*hh_block[current_last][i];
				}

				// diagonal element (X value)
				Xlocal[i][i-firstdiag]=Xlocal[i][i-firstdiag]*h_block[current_last][i];

				// before diagonal element (K values)
				for (j=(i-firstdiag)-1; j>=0; j--)
				{
					Klocal[i][j]=Klocal[i][j]*h_block[current_last][i] - Klocal[l][j]*hh_block[current_last][i];
				}
			}
			// before first diagonal element
			for (i=firstdiag-1; i>=0; i--)
			{//columns:
				//printf("AA (%d) %d:%d\n",rank,l,i);

				// after column l (X values)
				for (j=mycols-1; j>l_col; j--)
				{
					Xlocal[i][j]=Xlocal[i][j]*h_block[current_last][i] - Xlocal[l][j]*hh_block[current_last][i];
				}

				// column l (X value)
				Xlocal[i][l_col]= - Xlocal[l][l_col]*hh_block[current_last][i];

				// before column l (K values)
				for (j=l_col-1; j>=0; j--)
				{
					Klocal[i][j]=Klocal[i][j]*h_block[current_last][i] - Klocal[l][j]*hh_block[current_last][i];
				}
			}
		}
		else										// proc. NOT containing column l
		{//rows:
			int very_last_row;
			very_last_row=MIN((firstdiag+mycols),l);

			// remaining
			for (i=l-1; i>=very_last_row; i--)
			{
				//printf("C (%d) %d:%d\n",rank,l,i);

				// after column l (X values)
				for (j=mycols-1; j>=l_col; j--)
				{
					Xlocal[i][j]=Xlocal[i][j]*h_block[current_last][i] - Xlocal[l][j]*hh_block[current_last][i];
				}

				// before column l (K values)
				for (j=l_col-1; j>=0; j--)
				{
					Klocal[i][j]=Klocal[i][j]*h_block[current_last][i] - Klocal[l][j]*hh_block[current_last][i];
				}
				//TODO: loop join
			}
			// containing first diagonal element
			for (i=very_last_row-1; i>=firstdiag; i--)
			{//columns:
				//printf("B (%d) %d:%d\n",rank,l,i);

				// after diagonal element (K values)
				for (j=mycols-1; j>i-firstdiag; j--)
				{
					Klocal[i][j]=Klocal[i][j]*h_block[current_last][i] - Klocal[l][j]*hh_block[current_last][i];
				}

				// diagonal element (X value)
				Xlocal[i][i-firstdiag]=Xlocal[i][i-firstdiag]*h_block[current_last][i];

				// before diagonal element (K values)
				for (j=(i-firstdiag)-1; j>=0; j--)
				{
					Klocal[i][j]=Klocal[i][j]*h_block[current_last][i] - Klocal[l][j]*hh_block[current_last][i];
				}
			}
			// before first diagonal element
			for (i=MIN(firstdiag,l)-1; i>=0; i--)
			{//columns:
				//printf("A (%d) %d:%d\n",rank,l,i);

				// after column l (X values)
				for (j=mycols-1; j>=l_col; j--)
				{
					Xlocal[i][j]=Xlocal[i][j]*h_block[current_last][i] - Xlocal[l][j]*hh_block[current_last][i];
				}

				// before column l (K values)
				for (j=l_col-1; j>=0; j--)
				{
					Klocal[i][j]=Klocal[i][j]*h_block[current_last][i] - Klocal[l][j]*hh_block[current_last][i];
				}
				//TODO: loop join
			}
		}

		///////// update local copy of global last rows and cols of K
		if (current_last>0) // block of last rows (cols) not completely scanned
		{
			// useless because already done in block at the beginning
			/*
			for (i=0;i<current_last;i++)
			{
				for (j=0; j<=l-1; j++)
				{
					lastKc[i][j]=lastKc[i][j]*h_block[current_last][j]   - lastKr[current_last][l-current_last+i]*hh_block[current_last][j];
					lastKr[i][j]=lastKr[i][j]*h_block[current_last][l-current_last+i] - lastKr[current_last][j]*hh_block[current_last][l-current_last+i];
				}
			}
			*/
			current_last--; // update counter to point to the "future" last row (col) in the block
		}
		else // block of last rows (cols) completely scanned
		{
			//TODO: re-shuffle lines for a minimal ovarlap calc/comm
			current_last=nb-1; // reset counter for next block (to be sent/received)

			l_rank_future = PVMAP(l-nb, mycols);

			// collect chunks of last row of K to "future" last node
			// "current" last node sends smaller chunks until 0
			MPI_Iallgatherv (&Klocal[l-nb][0], gather_count[rank], s_lastKr_chunk_resized, &lastKr[0][0], gather_count, gather_displacement, r_lastKr_chunk_resized, comm, &mpi_request[0]);


			//future last node broadcasts last rows and cols of K
			if (rank==l_rank_future)
			{
				// copy data into local buffer before broadcast
				for(j=0;j<nb;j++)
				{
					for (i=0; i<=l-1; i++)
					{
						lastKc[j][i]=Klocal[i][PVLOCAL(l-nb, mycols)+j];
					}
				}
			}
			// do not wait all for gather: only who has to broadcast
			//MPI_Wait(&mpi_request, &mpi_status);
			MPI_Ibcast (&lastKc[0][0], l-1, lastKc_col_resized, l_rank_future, comm, &mpi_request[1]);

			// decrease the size of next chunk from "current" last node
			gather_count[l_rank_future]=gather_count[l_rank_future]-nb;
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

	MPI_Waitall(2, mpi_request, mpi_status);
	MPI_Barrier(MPI_COMM_WORLD);

	// cleanup
	NULLFREE(lastKc);
	NULLFREE(lastKr);
	NULLFREE(gather_displacement);
	NULLFREE(gather_count);

	DeallocateMatrix2D(lastK,2*nb,CONTIGUOUS);
	DeallocateMatrix2D(h_block,nb,CONTIGUOUS);
	DeallocateMatrix2D(hh_block,nb,CONTIGUOUS);
	DeallocateMatrix2D(Tlocal,n,CONTIGUOUS);

	result.total_end_time = time(NULL);

	return result;
}
