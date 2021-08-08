#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "helpers/macros.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "DGEZR.h"
#include "pbDGEIT_CX.bf1.ftx.h"
#include "pbDGEUT_CO.h"
#include "pbDGEUX_CO.h"
#include "pbDGEUH_CO.h"
#include "pbDGEUB_CO.h"

test_output pbDGESV_CO_bf1_ftx(double** A, double** bb, double** xx, test_input input, parallel_env env)
{
	/*
	 * nb	NOT USED: blocking factor: number of adjacent column (block width)
	 * n	size (number of columns) of the square matrix A
	 * m	number of rigth-hand-sides (number of columns) in bb
	 *
	 */

	test_output result;

	result.total_start_time = time(NULL);


	MPI_Comm comm_calc;
	MPI_Comm comm_row;
	MPI_Comm comm_row_calc;
	MPI_Comm comm_col;

    int cprocrows;
    int cproccols;
    int sprocrows;
    int sproccols;
    int mpi_row;
    int mpi_col;

    cprocrows = sqrt(env.calc_procs);
    cproccols = cprocrows;

    sprocrows = cprocrows;
    sproccols = env.spare_procs / sprocrows;

	if ( likely(env.mpi_rank < env.calc_procs) )
	{
		MPI_Comm_split(MPI_COMM_WORLD, 0, env.mpi_rank, &comm_calc);	// communicator for calc only nodes

	    mpi_row = env.mpi_rank / cproccols;					// put nodes on grid
	    mpi_col = env.mpi_rank % cproccols;
	}
	else
	{
		MPI_Comm_split(MPI_COMM_WORLD, 1, env.mpi_rank, &comm_calc);	// communicator for spare only nodes

	    mpi_row = (env.mpi_rank-env.calc_procs) / sproccols;
	    mpi_col = cproccols + ((env.mpi_rank-env.calc_procs) % sproccols);
	}
	MPI_Comm_split(comm_calc, mpi_row, env.mpi_rank, &comm_row_calc);// communicator for a row of calc (or spare) only nodes
	MPI_Comm_split(MPI_COMM_WORLD, mpi_row, env.mpi_rank, &comm_row); 		// communicator for a row
	MPI_Comm_split(MPI_COMM_WORLD, mpi_col, env.mpi_rank, &comm_col);			// communicator for a col

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

	int i,l,gi;							// general indexes
    int rhs;							// r.h.s. index
    int myrows   = input.n/cprocrows;	// num of rows per process
    int mycols   = input.n/cproccols;	// num of cols per process
    int myxxrows = mycols;				// num of chunks for better code readability

    /*
     * check grid
     */
    /*
    for (i=0; i<env.calc_procs+env.spare_procs; i++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		if (env.mpi_rank==i)
		{
			printf("%d@(%d,%d) = (%d,%d)\n",env.mpi_rank,mpi_row,mpi_col,mpi_rank_row_in_col,mpi_rank_col_in_row);

			fflush(stdout);
		}
	}
	*/

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Tlocal;
			 Tlocal=AllocateMatrix2D(myrows, mycols, CONTIGUOUS);

	// last row and col of K
	double** lastK;
			 lastK=AllocateMatrix2D(2, mycols, CONTIGUOUS);
	// aliases
	double*  lastKr=&lastK[0][0];
	double*  lastKc=&lastK[1][0];

	// helper vectors
    double* h;
    		h=AllocateVector(myrows);


	/*
	 *  init inhibition table
	 */
	pbDGEIT_CX_bf1_ft(A, Tlocal, lastKr, lastKc, input.n, env.calc_procs,
			sproccols,
			comm_calc, env.mpi_rank,
			comm_row, mpi_rank_col_in_row,
			comm_col, mpi_rank_row_in_col,
			comm_row_calc,
			mpi_status,
			mpi_request
			);											// init inhibition table

	if (mpi_rank_row_in_col == 0 && mpi_rank_col_in_row < cproccols) 							// first row of calc procs
	{
		MPI_Ibcast (&bb[0][0], input.n*input.nrhs, MPI_DOUBLE, 0, comm_row_calc, mpi_req_bb);	// send all r.h.s
		DGEZR(xx, input.n, input.nrhs);															// init (zero) solution vectors
	}

		/*
		 * check initialization
		 */
		/*
		MPI_Waitall(4, mpi_request, mpi_status);

		for (i=0; i<env.calc_procs+env.spare_procs; i++)
		{
			MPI_Barrier(MPI_COMM_WORLD);
			if (env.mpi_rank==i)
			{
				printf("%d@(%d,%d)\n",input.n,mpi_rank_row_in_col,mpi_rank_col_in_row);
				PrintMatrix2D(Tlocal, myrows, mycols);
				printf("\n");
				PrintMatrix2D(lastK, 2, mycols);

				fflush(stdout);
			}
		}
		*/


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

		if ( mpi_rank_col_in_row < cproccols )
		{
			if ( mpi_rank_row_in_col == 0 ) 					// first row of calc procs
			{
				// all levels but last one (l=0)
				for (l=input.n-1; l>0; l--)
				{
					l_owner  = PVMAP(l, myrows);
					l_row    = PVLOCAL(l, myrows);
					l_col    = l_row;

					l_1_owner = PVMAP(l-1, mycols);
					l_1_col   = PVLOCAL(l-1, mycols);
					l_1_row   = l_1_col;

					// last_row = myrows | l_row | 0
					last_row = ( mpi_rank_row_in_col < l_owner )*myrows + ( mpi_rank_row_in_col == l_owner )*l_row;

					/*
					 * update solutions
					 */
					MPI_Waitall(2, mpi_request, mpi_status);	// wait for lastKr and bb
					pbDGEUX_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, input.nrhs,
									l, l_owner, l_1_owner, l_row, l_col,
									&myxxstart,
									lastKr, xx, bb);

					/*
					 * update helpers
					 */
					if ( mpi_rank_row_in_col == mpi_rank_col_in_row )
					{
						MPI_Waitall(3, mpi_req_row, mpi_st_row); // wait for lastKr and lastKc
						pbDGEUH_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, input.nrhs,
										l, l_owner, l_1_owner, l_row, l_col,
										last_row,
										lastKr, lastKc, h);
					}
					MPI_Ibcast ( &h[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row, mpi_req_h);

					/*
					 * update auxiliary causes
					 */
					pbDGEUB_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, input.nrhs,
									l, l_owner, l_1_owner, l_row, l_col,
									l_1_col,
									lastKr, bb);
					MPI_Ibcast (&bb[l-1][0], input.nrhs, MPI_DOUBLE, l_1_owner, comm_row_calc, mpi_req_bb);

					/*
					 * update table
					 */
					MPI_Waitall(3, mpi_req_row, mpi_st_row); // wait for lastKr, lastKc and h
					pbDGEUT_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, input.nrhs,
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
							#pragma ivdep
							for (i=0; i < mycols; i++)			// copy (l-1)-th row in buffer
							{
								lastKr[i]=Tlocal[l_1_row][i];
							}
							*/
							lastKr=Tlocal[l_1_row];
						}
						MPI_Ibcast ( &lastKr[0], mycols, MPI_DOUBLE, l_1_owner, comm_col, mpi_req_row);

						// sync last col
						if (mpi_rank_row_in_col <= l_1_owner)			// procs rows, including that one holding the (l-1)-th row
						{
							if (mpi_rank_col_in_row == l_1_owner)	// proc in the row that has (l-1)-th col
							{
								#pragma ivdep
								for (i=0; i < last_row; i++)	// copy (l-1)-th col in buffer
								{
									lastKc[i]=Tlocal[i][l_1_col];
								}
							}
							MPI_Ibcast ( &lastKc[0], last_row, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_col);
						}
				}// end of loop over levels
			}
			else
			{
				// all levels but last one (l=0)
				for (l=input.n-1; l>0; l--)
				{
					l_owner  = PVMAP(l, myrows);
					l_row    = PVLOCAL(l, myrows);
					l_col    = l_row;

					l_1_owner = PVMAP(l-1, mycols);
					l_1_col   = PVLOCAL(l-1, mycols);
					l_1_row   = l_1_col;

					// last_row = myrows | l_row | 0
					last_row = ( mpi_rank_row_in_col < l_owner )*myrows + ( mpi_rank_row_in_col == l_owner )*l_row;

					/*
					 * update helpers
					 */
					if ( mpi_rank_row_in_col == mpi_rank_col_in_row )
					{
						MPI_Waitall(3, mpi_req_row, mpi_st_row); // wait for lastKr and lastKc
						pbDGEUH_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, input.nrhs,
										l, l_owner, l_1_owner, l_row, l_col,
										last_row,
										lastKr, lastKc, h);
					}
					MPI_Ibcast ( &h[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row, mpi_req_h);

					/*
					 * update table
					 */
					MPI_Waitall(3, mpi_req_row, mpi_st_row); // wait for lastKr, lastKc and h
					pbDGEUT_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, input.nrhs,
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
							#pragma ivdep
							for (i=0; i < mycols; i++)			// copy (l-1)-th row in buffer
							{
								lastKr[i]=Tlocal[l_1_row][i];
							}
							*/
							lastKr=Tlocal[l_1_row];
						}
						MPI_Ibcast ( &lastKr[0], mycols, MPI_DOUBLE, l_1_owner, comm_col, mpi_req_row);

						// sync last col
						if (mpi_rank_row_in_col <= l_1_owner)			// procs rows, including that one holding the (l-1)-th row
						{
							if (mpi_rank_col_in_row == l_1_owner)	// proc in the row that has (l-1)-th col
							{
								#pragma ivdep
								for (i=0; i < last_row; i++)	// copy (l-1)-th col in buffer
								{
									lastKc[i]=Tlocal[i][l_1_col];
								}
							}
							MPI_Ibcast ( &lastKc[0], last_row, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_col);
						}
				}// end of loop over levels
			}
		}
		else
		{
			// all levels but last one (l=0)
			for (l=input.n-1; l>0; l--)
			{
				l_owner  = PVMAP(l, myrows);
				l_row    = PVLOCAL(l, myrows);
				l_col    = l_row;

				l_1_owner = PVMAP(l-1, mycols);
				l_1_col   = PVLOCAL(l-1, mycols);
				l_1_row   = l_1_col;

				// last_row = myrows | l_row | 0
				last_row = ( mpi_rank_row_in_col < l_owner )*myrows + ( mpi_rank_row_in_col == l_owner )*l_row;

				/*
				 * update helpers
				 */
				MPI_Ibcast ( &h[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row, mpi_req_h);

				/*
				 * update table
				 */
				MPI_Waitall(3, mpi_req_row, mpi_st_row); // wait for lastKr, lastKc and h
				pbDGEUT_CO (	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, input.nrhs,
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
						#pragma ivdep
						for (i=0; i < mycols; i++)			// copy (l-1)-th row in buffer
						{
							lastKr[i]=Tlocal[l_1_row][i];
						}
						*/
						lastKr=Tlocal[l_1_row];
					}
					MPI_Ibcast ( &lastKr[0], mycols, MPI_DOUBLE, l_1_owner, comm_col, mpi_req_row);

					// sync last col
					if (mpi_rank_row_in_col <= l_1_owner)			// procs rows, including that one holding the (l-1)-th row
					{
						if (mpi_rank_col_in_row == l_1_owner)	// proc in the row that has (l-1)-th col
						{
							#pragma ivdep
							for (i=0; i < last_row; i++)	// copy (l-1)-th col in buffer
							{
								lastKc[i]=Tlocal[i][l_1_col];
							}
						}
						MPI_Ibcast ( &lastKc[0], last_row, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_col);
					}
			}// end of loop over levels
		}

		/*
		 * update last level of the table
		 */
			// last level (l=0)
			if ( mpi_rank_row_in_col == 0 && mpi_rank_col_in_row < cproccols) 								// first row of calc procs
			{
				// bb[0] must be here
				MPI_Wait( mpi_req_bb, mpi_st_bb);

				#pragma ivdep
				for (i=0; i<myxxrows; i++)
				{
					gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
					for(rhs=0;rhs<input.nrhs;rhs++)
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
					MPI_Gather (MPI_IN_PLACE, input.nrhs*myxxrows, MPI_DOUBLE, &xx[0][0], input.nrhs*myxxrows, MPI_DOUBLE, 0, comm_row_calc);
				}
				else
				{
					MPI_Gather (&xx[mpi_rank_col_in_row*myxxrows][0], input.nrhs*myxxrows, MPI_DOUBLE, &xx[0][0], input.nrhs*myxxrows, MPI_DOUBLE, 0, comm_row_calc);
				}
			}
			else
			{
				result.core_end_time = time(NULL);

				// TODO: add checking on exit code
				result.exit_code = 0;
			}

			/*
			 * check table result
			 */
			/*
			MPI_Barrier(MPI_COMM_WORLD);
			if ( mpi_rank_row_in_col == 0 ) 								// first row of calc procs
			{
				for (i=0; i<cproccols+sproccols; i++)
				{
					MPI_Barrier(comm_row);
					if (mpi_rank_col_in_row==i)
					{
						printf("l=0 @(%d,%d):\n",mpi_rank_row_in_col,mpi_rank_col_in_row);
						PrintMatrix2D(Tlocal, myrows, mycols);
						printf("\n");

						fflush(stdout);
					}
					MPI_Barrier(comm_row);
				}
			}
			*/

	// wait to complete before cleanup and return
	MPI_Waitall(4, mpi_request, mpi_status);

	// cleanup
	DeallocateMatrix2D(lastK,2,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateMatrix2D(Tlocal,myrows,CONTIGUOUS);

	result.total_end_time = time(NULL);

	return result;
}

