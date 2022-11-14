#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "helpers/macros.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "DGEZR.h"
#include "pbDGEIT_CX.bf1.h"


test_output pbDGESV_WO_bf1(int nb, int n, double** A, int m, double** bb, double** xx, MPI_Comm comm)
{
	/*
	 * nb	NOT USED: blocking factor: number of adjacent column (block width)
	 * n	size (number of columns) of the square matrix A
	 * m	number of rigth-hand-sides (number of columns) in bb
	 *
	 */

	test_output result;

	result.total_start_time = time(NULL);

    int mpi_rank, cprocs;
    MPI_Comm_rank(comm, &mpi_rank);	// get current process id
    MPI_Comm_size(comm, &cprocs);	// get number of processes

    int cprocrows = sqrt(cprocs);
    int cproccols = cprocrows;		// cols == rows, but better code readability

    int mpi_row = mpi_rank / cproccols;
    int mpi_col = mpi_rank % cproccols;

	MPI_Comm comm_row;
	MPI_Comm comm_col;
	MPI_Comm_split(comm, mpi_row, mpi_rank, &comm_row);
	MPI_Comm_split(comm, mpi_col, mpi_rank, &comm_col);

	int mpi_rank_col_in_row;
	int mpi_rank_row_in_col;
	MPI_Comm_rank(comm_row, &mpi_rank_col_in_row);		// get current process id in row
	MPI_Comm_rank(comm_col, &mpi_rank_row_in_col);		// get current process id in col

	MPI_Status  mpi_status[4];
	MPI_Request mpi_request[4];
				mpi_request[0] = MPI_REQUEST_NULL;
				mpi_request[1] = MPI_REQUEST_NULL;
				mpi_request[2] = MPI_REQUEST_NULL;
				mpi_request[3] = MPI_REQUEST_NULL;
	//aliases
	MPI_Request* mpi_req_bb  = &mpi_request[0]; // req. for bb broadcast
	MPI_Request* mpi_req_row = &mpi_request[1]; // req. for row broadcast
	MPI_Request* mpi_req_h   = &mpi_request[2]; // req. for h broadcast
	MPI_Request* mpi_req_col = &mpi_request[3]; // req. for col broadcast

	MPI_Status* mpi_st_bb  = &mpi_status[0];
	MPI_Status* mpi_st_row = &mpi_status[1];
	//MPI_Status* mpi_st_h   = &mpi_status[2]; // never referenced explicitly
	//MPI_Status* mpi_st_col = &mpi_status[3]; // never referenced explicitly

	int i,j,l,gi;					// general indexes
    int rhs;						// r.h.s. index
    int myrows   = n/cprocrows;		// num of rows per process
    int mycols   = n/cproccols;		// num of cols per process
    int myxxrows = mycols;			// num of chunks for better code readability
    int bf       = 1;				// blocking factor

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Klocal;
			 Klocal=AllocateMatrix2D(myrows, mycols, CONTIGUOUS);

	double** Xlocal;
			 Xlocal=AllocateMatrix2D(myrows, mycols, CONTIGUOUS);

	// last rows and cols of K
	double** lastK;
			 lastK=AllocateMatrix2D(2*bf, mycols, CONTIGUOUS);	// last rows [0 - (bf-1)] and cols [ bf - (2bf -1)] of K
	// aliases
	double*  lastKr=&lastK[0][0];
	double*  lastKc=&lastK[1][0];

	// helper vectors
    double* h;
    		h=AllocateVector(myrows);
    double hh;

	/*
	 *  init inhibition table
	 */
	DGEZR(xx, n, m);											// init (zero) solution vectors
	pbDGEIT_CX_bf1(A, Klocal, lastKr, lastKc, n, cprocs,		// uses compact table for initialization
			comm, mpi_rank, comm_row,
			mpi_rank_col_in_row, comm_col,
			mpi_rank_row_in_col,
			mpi_status,
			mpi_request
			);

	MPI_Waitall(4, mpi_request, mpi_status);					// wait for compact table ready

	if (mpi_rank_row_in_col==mpi_rank_col_in_row) 				// and then extract diagonal elements for the X table
	{
		for (i=0; i<myrows; i++)
		{
			for (j=0; j<i; j++)
			{
				Xlocal[i][j]=0;
			}

			Xlocal[i][i]=Klocal[i][i];

			for (j=i+1; j<mycols; j++)
			{
				Xlocal[i][j]=0;
			}
		}
	}
	/* don't need to zero off-diagonal elements, because different per-case formulae assume it already
	 * could be a strategy to be tested: zero all, use una formula and vectorize
	 */
	/*
	{
		for (i=0; i<myrows; i++)
		{
			for (j=0; j<i; j++)
			{
				Xlocal[i][j]=0;
			}

			Xlocal[i][i]=Klocal[i][i];

			for (j=i+1; j<mycols; j++)
			{
				Xlocal[i][j]=0;
			}
		}
	}
	else
	{
		for (i=0; i<myrows; i++)
		{
			for (j=0; j<mycols; j++)
			{
				Xlocal[i][j]=0;
			}
		}
	}
	*/


	/*
	 * check initialization
	 */
		/*
		MPI_Waitall(4, mpi_request, mpi_status);

		for (i=0; i<cprocs; i++)
		{
			MPI_Barrier(MPI_COMM_WORLD);
			if (mpi_rank==i)
			{
				printf("%d@(%d,%d)\n",n,mpi_rank_row_in_col,mpi_rank_col_in_row);
				PrintMatrix2D(Xlocal, myrows, mycols);
				printf("\n");
				PrintMatrix2D(Klocal, myrows, mycols);
				printf("\n");
				PrintMatrix2D(lastK, 2, mycols);

				fflush(stdout);
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
		*/

	if (mpi_rank_row_in_col==0) 											// first row of procs
	{
		MPI_Ibcast (&bb[0][0], n*m, MPI_DOUBLE, 0, comm_row, mpi_req_bb);	// send all r.h.s
	}


	/*
	 *  calc inhibition sequence
	 */
	result.core_start_time = time(NULL);

	// general bounds for the loops over the columns
	// they differ on processes and change along the main loop over the levels
	int myxxstart = mycols;		// beginning column position for updating the solution (begins from right)
	int last_row;				// (local) last row to process
	int l_1_owner;				// proc rank hosting the (l-1)-th row (or col)
	int l_1_row;				// local index of the (l-1)-th row
	int l_1_col;				// local index of the (l-1)-th col
	int l_owner;				// proc rank hosting the l-th row (or col)
	int l_row;					// local index of the l-th row
	int l_col;					// local index of the l-th col

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		l_owner  = PVMAP(l, myrows);
		l_row    = PVLOCAL(l, myrows);
		l_col    = l_row;

		l_1_owner = PVMAP(l-1, mycols);
		l_1_col   = PVLOCAL(l-1, mycols);
		l_1_row   = l_1_col;

		/*
		 * update solutions
		 */
			if (mpi_rank_row_in_col==0) 					// first row of procs
			{
				// if a process contains the l-th cols, must skip it
				// if (mpi_rank_col_in_row==l_owner) {myxxstart--;}
				// avoid if:
				myxxstart = myxxstart - (mpi_rank_col_in_row == l_owner);

				MPI_Waitall(2, mpi_request, mpi_status); // wait for lastKr and bb

				// update xx vector
				for (i=myxxstart; i<myrows; i++)
				{
					gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
					for (rhs=0;rhs<m;rhs++)
					{
						xx[gi][rhs]=xx[gi][rhs]+lastKr[i]*bb[l][rhs];
					}
				}
			}


		/*
		 * update helpers
		 */
			// chunks of last row and col are symmetrical to the main diagonal
			// they are hosted on procs on the diagonal
			// last_row = myrows | l_row | 0
			last_row = ( mpi_rank_row_in_col < l_owner )*myrows + ( mpi_rank_row_in_col == l_owner )*l_row;

			if (mpi_rank_row_in_col==mpi_rank_col_in_row)
			{
				MPI_Waitall(3, mpi_req_row, mpi_st_row); // wait for lastKr and lastKc

				#pragma GCC ivdep
				for (i=0; i<last_row; i++)
				{
					h[i]   = 1/(1-lastKc[i]*lastKr[i]);
				}
			}
			MPI_Ibcast ( &h[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row, mpi_req_h);

		/*
		 * update auxiliary causes
		 */
			if (mpi_rank_row_in_col==0) 							// first row of procs
			{
				// no need to "MPI_Wait": lastKr is already there

				// TODO: remove ifs
				if ( mpi_rank_col_in_row < l_1_owner ) 			// procs before col l-1
				{
					for (i=0; i<mycols; i++)				// treat all cols
					{
						gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
						for (rhs=0;rhs<m;rhs++)
						{
							bb[gi][rhs] = bb[gi][rhs]-lastKr[i]*bb[l][rhs];
						}
					}
				}
				else if ( mpi_rank_col_in_row == l_1_owner ) 		// proc holding col l-1
				{
					for (i=0; i<=l_1_col; i++)					// cols till l-1
					{
						gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
						for (rhs=0;rhs<m;rhs++)
						{
							bb[gi][rhs] = bb[gi][rhs]-lastKr[i]*bb[l][rhs];
						}
					}
				}
				//else	{do nothing}								// procs after col l-1

				// send l-1 values to other procs for next iteration
				MPI_Ibcast (&bb[l-1][0], m, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_bb);
			}


		/*
		 * update table
		 */
			// last row is computed above
			// for procs with last_row == 0, nothing to do

			MPI_Waitall(3, mpi_req_row, mpi_st_row); // wait for lastKr, lastKc and h

			if (mpi_rank_col_in_row == l_owner)						// proc holding the l-th col AND ..
			{
				if (mpi_rank_col_in_row == mpi_rank_row_in_col)		// proc holding the l-th col AND ON the diagonal
				{
					for (i=0; i < last_row; i++)
					{
						hh  = lastKc[i]*h[i];

						// before diagonal
						#pragma GCC ivdep
						for (j=0; j < i; j++)
						{
							Klocal[i][j] = Klocal[i][j]*h[i] - lastKr[j]*hh;
						}

						// on diagonal
						Xlocal[i][j] = Xlocal[i][j]*h[i];

						// after diagonal but before l-th
						#pragma GCC ivdep
						for (j=i+1; j < l_col; j++)
						{
							Klocal[i][j] = Klocal[i][j]*h[i] - lastKr[j]*hh;
						}

						// on l-th
						Xlocal[i][j] = - lastKr[j]*hh;

						// after l-th
						#pragma GCC ivdep
						for (j=l_col+1; j < mycols; j++)
						{
							Xlocal[i][j] = Xlocal[i][j]*h[i] - lastKr[j]*hh;
						}
					}
				}
				else												// proc holding the l-th col AND OFF the diagonal
				{
					for (i=0; i < last_row; i++)
					{
						hh  = lastKc[i]*h[i];

						// before l-th
						#pragma GCC ivdep
						for (j=0; j < l_col; j++)
						{
							Klocal[i][j] = Klocal[i][j]*h[i] - lastKr[j]*hh;
						}

						// on l-th
						Xlocal[i][j] = - lastKr[j]*hh;

						// after l-th
						#pragma GCC ivdep
						for (j=l_col+1; j < mycols; j++)
						{
							Xlocal[i][j] = Xlocal[i][j]*h[i] - lastKr[j]*hh;
						}
					}
				}
			}
			else												// proc NOT holding the l-th col AND ..
			{
				if (mpi_rank_col_in_row == mpi_rank_row_in_col)	// proc NOT holding the l-th col AND ON the diagonal
				{
					for (i=0; i < last_row; i++)
					{
						hh  = lastKc[i]*h[i];

						// before diagonal
						#pragma GCC ivdep
						for (j=0; j < i; j++)
						{
							Klocal[i][j] = Klocal[i][j]*h[i] - lastKr[j]*hh;
						}

						// on diagonal
						Xlocal[i][j] = Xlocal[i][j]*h[i];

						// after diagonal
						#pragma GCC ivdep
						for (j=i+1; j < mycols; j++)
						{
							Klocal[i][j] = Klocal[i][j]*h[i] - lastKr[j]*hh;
						}
					}
				}
				else if (mpi_rank_col_in_row < mpi_rank_row_in_col)	// proc NOT holding the l-th col AND OFF AND BEFORE the diagonal
				{

					for (i=0; i < last_row; i++)
					{
						hh  = lastKc[i]*h[i];

						#pragma GCC ivdep
						for (j=0; j < mycols; j++)
						{
							Klocal[i][j] = Klocal[i][j]*h[i] - lastKr[j]*hh;
						}
					}
				}
				else												// proc NOT holding the l-th col AND OFF AND AFTER the diagonal
				{
					if (mpi_rank_col_in_row < l_owner)
					{
						for (i=0; i < last_row; i++)
						{
							hh  = lastKc[i]*h[i];

							#pragma GCC ivdep
							for (j=0; j < mycols; j++)
							{
								Klocal[i][j] = Klocal[i][j]*h[i] - lastKr[j]*hh;
							}
						}
					}
					else
					{
						for (i=0; i < last_row; i++)
						{
							hh  = lastKc[i]*h[i];

							#pragma GCC ivdep
							for (j=0; j < mycols; j++)
							{
								Xlocal[i][j] = Xlocal[i][j]*h[i] - lastKr[j]*hh;
							}
						}
					}

				}
			}


		/*
		 * sync future last (l-1) row and col
		 */
			// sync last row
			if (mpi_rank_row_in_col == l_1_owner)				// procs row that holds the (l-1)-th row AND ..
			{
				if (mpi_rank_col_in_row < mpi_rank_row_in_col)	// procs row that holds the (l-1)-th row AND BEFORE the diagonal				{
				{
					#pragma GCC ivdep
					for (i=0; i < mycols; i++)			// copy (l-1)-th row in buffer
					{
						lastKr[i]=Klocal[l_1_row][i];
					}
				}
				else if (mpi_rank_col_in_row > mpi_rank_row_in_col) // procs row that holds the (l-1)-th row AND AFTER the diagonal
				{
					#pragma GCC ivdep
					for (i=0; i < mycols; i++)			// copy (l-1)-th row in buffer
					{
						lastKr[i]=Xlocal[l_1_row][i];
					}
				}
				else												// procs row that holds the (l-1)-th row AND ON the diagonal
				{
					#pragma GCC ivdep
					for (i=0; i < l_1_col; i++)			// copy (l-1)-th row in buffer
					{
						lastKr[i]=Klocal[l_1_row][i];
					}
					#pragma GCC ivdep
					for (i=l_1_col; i < mycols; i++)			// copy (l-1)-th row in buffer
					{
						lastKr[i]=Xlocal[l_1_row][i];
					}
				}

			}
			MPI_Ibcast ( &lastKr[0], mycols, MPI_DOUBLE, l_1_owner, comm_col, mpi_req_row);

			// sync last col
			if (mpi_rank_row_in_col <= l_1_owner)			// procs rows, including that one holding the (l-1)-th row
			{
				if (mpi_rank_col_in_row == l_1_owner)	// proc in the row that has (l-1)-th col
				{
					#pragma GCC ivdep
					for (i=0; i < last_row; i++)	// copy (l-1)-th col in buffer
					{
						lastKc[i]=Klocal[i][l_1_col];
					}
				}
				MPI_Ibcast ( &lastKc[0], last_row, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_col);
			}

	}// end of loop over levels


	/*
	 * update last level of the table
	 */
		// last level (l=0)
		if (mpi_rank_row_in_col==0) 								// first row of procs
		{
			// bb[0] must be here
			MPI_Wait( mpi_req_bb, mpi_st_bb);

			for (i=0; i<myxxrows; i++)
			{
				gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
				for(rhs=0;rhs<m;rhs++)
				{
					xx[gi][rhs]=xx[gi][rhs]+Xlocal[0][i]*bb[0][rhs];
				}
			}

			result.core_end_time = time(NULL);

			// TODO: add checking on exit code
			result.exit_code = 0;

			// collect solution
			// MPI_IN_PLACE required for MPICH based versions
			if (mpi_rank_col_in_row==0)
			{
				MPI_Gather (MPI_IN_PLACE, m*myxxrows, MPI_DOUBLE, &xx[0][0], m*myxxrows, MPI_DOUBLE, 0, comm_row);
			}
			else
			{
				MPI_Gather (&xx[mpi_rank_col_in_row*myxxrows][0], m*myxxrows, MPI_DOUBLE, &xx[0][0], m*myxxrows, MPI_DOUBLE, 0, comm_row);
			}
		}
		else
		{
			result.core_end_time = time(NULL);

			// TODO: add checking on exit code
			result.exit_code = 0;
		}


	// wait to complete before cleanup and return
	MPI_Waitall(4, mpi_request, mpi_status);

	// cleanup
	DeallocateMatrix2D(lastK,2,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateMatrix2D(Xlocal,myrows,CONTIGUOUS);
	DeallocateMatrix2D(Klocal,myrows,CONTIGUOUS);

	result.total_end_time = time(NULL);

	return result;
}
