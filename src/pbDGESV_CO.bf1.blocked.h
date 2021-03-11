#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "helpers/macros.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "DGEZR.h"
#include "pbDGEIT_CX.bf1.h"
#include "pbDGEUT_CO.h"
#include "pbDGEUX_CO.h"
#include "pbDGEUH_CO.h"
#include "pbDGEUB_CO.h"

test_output pbDGESV_CO_dev(int nb, int n, double** A, int m, double** bb, double** xx, MPI_Comm comm)
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
	MPI_Status* mpi_st_h   = &mpi_status[2];
	MPI_Status* mpi_st_col = &mpi_status[3];

	int i,l;						// general indexes
    int rhs;						// r.h.s. index
    int myrows   = n/cprocrows;		// num of rows per process
    int mycols   = n/cproccols;		// num of cols per process
    int myxxrows = mycols;			// num of chunks for better code readability
    int bf       = mycols;			// blocking factor

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Tlocal;
			 Tlocal=AllocateMatrix2D(myrows, mycols, CONTIGUOUS);

	// last rows and cols of K
	double** lastK;
			 lastK=AllocateMatrix2D(2, mycols, CONTIGUOUS);	// last rows [0 - (bf-1)] and cols [ bf - (2bf -1)] of K
	// aliases
	double*  lastKr=&lastK[0][0];
	double*  lastKc=&lastK[1][0];

	// helper vectors
    double* h;
    		h=AllocateVector(myrows);
    //double hh;
    /*
    double* hh;
			hh=AllocateVector(myrows);
	*/

	/*
	 *  init inhibition table
	 */
	DGEZR(xx, n, m);											// init (zero) solution vectors
	pbDGEIT_CX_bf1(A, Tlocal, lastKr, lastKc, n, cprocs,
			comm, mpi_rank, comm_row,
			mpi_rank_col_in_row, comm_col,
			mpi_rank_row_in_col,
			mpi_status,
			mpi_request
			);											// init inhibition table

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
				PrintMatrix2D(Tlocal, myrows, mycols);
				printf("\n");
				PrintMatrix2D(lastK, 2, mycols);

				fflush(stdout);
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
		*/

	if (mpi_rank_row_in_col==0) 								// first row of procs
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

	int gi;			// global i
	int l_1_owner;	// proc rank hosting the (l-1)-th row (or col)
	int l_1_col;	// local index of the (l-1)-th col
	int l_1_row;
	int l_owner;	// proc rank hosting the l-th row (or col)
	int l_row;
	int l_col;

	int blocks_tot=n/bf;	// bf=myrows => blocks_tot=cprocrows
	int block;
	int lb;

	// all levels but last block of levels
	for (block=blocks_tot-1; block>0; block--)
	{
		// procs rows, including that one holding the (l-1)-th row
		if ( likely ( mpi_rank_row_in_col <= block) ) // only active processes
		{
			l_owner  = block; //TODO: avoidable? block?

			for (lb=bf-1; lb>=0; lb--)
			{
				l=block*bf+lb;

				//l_owner  = PVMAP(l, myrows); //TODO: avoidable? block?
				l_row    = PVLOCAL(l, myrows);
				l_col    = l_row;

				l_1_owner = PVMAP(l-1, mycols);
				l_1_col   = PVLOCAL(l-1, mycols);
				l_1_row   = l_1_col;

				last_row = ( mpi_rank_row_in_col < l_owner )*myrows + ( mpi_rank_row_in_col == l_owner )*l_row;

				if (mpi_rank_row_in_col==0) 					// first row of procs
				{
					/*
					 * update solutions
					 */
					MPI_Waitall(2, mpi_request, mpi_status);	// wait for lastKr and bb
					pbDGEUX_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, rhs, m,
									l, l_owner, l_1_owner, l_row, l_col,
									&myxxstart,
									lastKr, xx, bb);
					/*
					 * update auxiliary causes
					 */
					pbDGEUB_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, rhs, m,
									l, l_owner, l_1_owner, l_row, l_col,
									l_1_col,
									lastKr, bb);
					MPI_Ibcast (&bb[l-1][0], m, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_bb);
				}

				/*
				 * update helpers
				 */
				// TODO: move helpers to top
				if (mpi_rank_row_in_col==mpi_rank_col_in_row)	// procs on diagonal
				{
					MPI_Waitall(3, mpi_req_row, mpi_st_row);	// wait for lastKr and lastKc
					pbDGEUH_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, rhs, m,
									l, l_owner, l_1_owner, l_row, l_col,
									last_row,
									lastKr, lastKc, h);
				}
				MPI_Ibcast ( &h[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row, mpi_req_h);

				/*
				 * update auxiliary causes
				 */
				/*
				if (mpi_rank_row_in_col==0) 					// first row of procs
				{
					pbDGEUB_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, rhs, m,
									l, l_owner, l_1_owner, l_row, l_col,
									l_1_col,
									lastKr, bb);
					MPI_Ibcast (&bb[l-1][0], m, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_bb);
				}
				*/

				/*
				 * update table
				 */
				MPI_Waitall(3, mpi_req_row, mpi_st_row);		// wait for lastKr, lastKc and h
				pbDGEUT_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, rhs, m,
								l, l_owner, l_1_owner, l_row, l_col,
								last_row,
								lastKr, lastKc, h, Tlocal);

				/*
				 * sync future last (l-1) row and col
				 */
					// sync last row
					if (mpi_rank_row_in_col == l_1_owner)			// procs row that holds the (l-1)-th row
					{
						/*
						for (i=0; i < mycols; i++)			// copy (l-1)-th row in buffer
						{
							lastKr[i]=Tlocal[l_1_row][i];
						}
						*/
						lastKr=Tlocal[l_1_row];
					}
					MPI_Ibcast ( &lastKr[0], mycols, MPI_DOUBLE, l_1_owner, comm_col, mpi_req_row);

					// sync last col
					if (mpi_rank_col_in_row == l_1_owner)	// proc in the row that has (l-1)-th col
					{
						#pragma ivdep
						for (i=0; i < last_row; i++)	// copy (l-1)-th col in buffer
						{
							lastKc[i]=Tlocal[i][l_1_col];
						}
					}
					MPI_Ibcast ( &lastKc[0], last_row, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_col);

			}// end of loop over levels in a block
		}
		else
		{
			for (lb=bf-1; lb>=0; lb--)
			{
				l=block*bf+lb;
				l_1_owner = PVMAP(l-1, mycols);
				MPI_Ibcast ( &lastKr[0], mycols, MPI_DOUBLE, l_1_owner, comm_col, mpi_req_row);
			}
		}
	}// end of loop over all blocks but last one

	if ( likely ( mpi_rank_row_in_col==0 ) ) 					// first row of procs
	{
		for (lb=bf-1; lb>0; lb--) // loop over level but last one in last block
		{
			// being block=0, there are some simplifications in calculating indexes
			//l=block*bf+lb;
			l=lb;
			//l_owner  = PVMAP(l, myrows);
			l_owner  = 0;
			l_row    = PVLOCAL(l, myrows);
			l_col    = l_row;

			//l_1_owner = PVMAP(l-1, mycols);
			l_1_owner = 0;
			l_1_col   = PVLOCAL(l-1, mycols);
			l_1_row   = l_1_col;

			//last_row = ( mpi_rank_row_in_col < l_owner )*myrows + ( mpi_rank_row_in_col == l_owner )*l_row;
			last_row = l_row;
			/*
			 * update solutions
			 */
			MPI_Wait(mpi_req_bb, mpi_st_bb);	// wait for bb
			pbDGEUX_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, rhs, m,
							l, l_owner, l_1_owner, l_row, l_col,
							&myxxstart,
							lastKr, xx, bb);

			/*
			 * update helpers
			 */
			if (mpi_rank_row_in_col==mpi_rank_col_in_row)
			{
				MPI_Wait(mpi_req_col, mpi_st_col);	// wait for lastKc
				pbDGEUH_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, rhs, m,
								l, l_owner, l_1_owner, l_row, l_col,
								last_row,
								lastKr, lastKc, h);
			}
			MPI_Ibcast ( &h[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row, mpi_req_h);

			/*
			 * update auxiliary causes
			 */
			pbDGEUB_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, rhs, m,
							l, l_owner, l_1_owner, l_row, l_col,
							l_1_col,
							lastKr, bb);
			MPI_Ibcast (&bb[l-1][0], m, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_bb);

			/*
			 * update table
			 */
			MPI_Waitall(2, mpi_req_h, mpi_st_h);		// wait for h and lastKc
			pbDGEUT_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, rhs, m,
							l, l_owner, l_1_owner, l_row, l_col,
							last_row,
							lastKr, lastKc, h, Tlocal);

			/*
			 * sync future last (l-1) row and col
			 */
				// no need to sync last row (already on proc row)
				/* if (mpi_rank_row_in_col == l_1_owner)			// procs row that holds the (l-1)-th row
				{
					for (i=0; i < mycols; i++)			// copy (l-1)-th row in buffer
					{
						lastKr[i]=Tlocal[l_1_row][i];
					}
				}
				MPI_Ibcast ( &lastKr[0], mycols, MPI_DOUBLE, l_1_owner, comm_col, mpi_req_row);
				*/
				lastKr=Tlocal[l_1_row];


				// sync last col
				if (mpi_rank_col_in_row == l_1_owner)	// proc in the row that has (l-1)-th col
				{
					#pragma ivdep
					for (i=0; i < last_row; i++)	// copy (l-1)-th col in buffer
					{
						lastKc[i]=Tlocal[i][l_1_col];
					}
				}
				MPI_Ibcast ( &lastKc[0], last_row, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_col);

		}// end of loop over levels in the last block
	}
	/*
	 * update last level of the table
	 */
		// last level (l=0)
		if (mpi_rank_row_in_col==0) 								// first row of procs
		{
			// bb[0] must be here
			MPI_Wait( mpi_req_bb, mpi_st_bb);

			#pragma ivdep
			for (i=0; i<myxxrows; i++)
			{
				gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
				for(rhs=0;rhs<m;rhs++)
				{
					xx[gi][rhs]=xx[gi][rhs]+Tlocal[0][i]*bb[0][rhs];
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
	DeallocateMatrix2D(Tlocal,myrows,CONTIGUOUS);

	result.total_end_time = time(NULL);

	return result;
}
