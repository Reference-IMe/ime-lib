#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "helpers/macros.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "DGEZR.h"
#include "pbDGEIT_CX.h"


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
	//MPI_Status* mpi_st_h   = &mpi_status[2]; // never referenced explicitly
	//MPI_Status* mpi_st_col = &mpi_status[3]; // never referenced explicitly

	int i,j,l;						// general indexes
    int rhs;						// r.h.s. index
    int myrows   = n/cprocrows;		// num of rows per process
    int mycols   = n/cproccols;		// num of cols per process
    int myxxrows = mycols;			// num of chunks for better code readability
    int bf           = nb;			// blocking factor
	//int current_last = bf-1;		// index for the current last row or col of K in buffer
    //TODO: use nb as blocking factor

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Tlocal;
			 Tlocal=AllocateMatrix2D(myrows, mycols, CONTIGUOUS);

	// last rows and cols of K
	double** lastK;
			 lastK=AllocateMatrix2D(2*bf, mycols, CONTIGUOUS);	// last rows [0 - (bf-1)] and cols [ bf - (2bf -1)] of K

	// aliases
	double** lastKr;
				lastKr=malloc(bf*sizeof(double*));
				for(i=0;i<bf;i++)
				{
					lastKr[i]=lastK[i];						// alias for last row
				}
	double** lastKc;
				lastKc=malloc(bf*sizeof(double*));
				for(i=0;i<bf;i++)
				{
					lastKc[i]=lastK[bf+i];					// alias for last col
				}

	// helper vectors
    double* h;
    		h=AllocateVector(myrows);
    double hh;

    double* hhh;
			hhh=AllocateVector(myrows);


	/*
	 *  init inhibition table
	 */
	DGEZR(xx, n, m);											// init (zero) solution vectors
	pbDGEIT_CX(A, Tlocal, lastK, n, bf, cprocs,
			comm, mpi_rank, comm_row,
			mpi_rank_col_in_row, comm_col,
			mpi_rank_row_in_col,
			mpi_status,
			mpi_request
			);											// init inhibition table

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
	//TODO: pre-calc other values..

	int gi;			// global i
	int l_1_owner;	// proc rank hosting the (l-1)-th row (or col)
	int l_1_col;	// local index of the (l-1)-th col
	int l_1_row;
	int l_owner;	// proc rank hosting the l-th row (or col)
	int l_row;
	int l_col;

	int blocks_tot=n/bf;
	int block;
	int lb;

	// all levels but last block of levels
	for (block=blocks_tot-1; block>0; block--)
	{
		for (lb=bf-1; lb>=0; lb--)
		{
			l=block*bf+lb;
			if (mpi_rank==0) printf("level %d\n",l);
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
					/*
					if (mpi_rank_col_in_row==l_owner)
					{
						myxxstart--;
					}
					*/
					// avoid if
					myxxstart = myxxstart - (mpi_rank_col_in_row == l_owner);

					// bb[l] must be here
					//MPI_Wait( mpi_req_bb, mpi_st_bb);

					// lastKr must be here
					//MPI_Wait( mpi_req_row, mpi_st_row);

					// bb[l] (or full bb, for the first iteration after init) and lastKr must be here
					MPI_Waitall(2, mpi_request, mpi_status);

					// update xx vector
					// l .. n-1
					for (i=myxxstart; i<myrows; i++)
					{
						gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
						for (rhs=0;rhs<m;rhs++)
						{
							xx[gi][rhs]=xx[gi][rhs]+lastKr[lb][i]*bb[l][rhs];
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
					// lastKc must be here
					//MPI_Wait( mpi_req_col, mpi_st_col);

					// lastKr must be here
					//MPI_Wait( mpi_req_row, mpi_st_row);

					MPI_Waitall(3, mpi_req_row, mpi_st_row);

	#pragma ivdep
					for (i=0; i<last_row; i++)
					{
						h[i]   = 1/(1-lastKc[lb][i]*lastKr[lb][i]);
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
								bb[gi][rhs] = bb[gi][rhs]-lastKr[lb][i]*bb[l][rhs];
							}
						}
					}
					else if ( mpi_rank_col_in_row == l_1_owner ) 		// proc holding col l-1
					{
						for (i=0; i<=l_1_col; i++)				// cols till l-1
						{
							gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
							for (rhs=0;rhs<m;rhs++)
							{
								bb[gi][rhs] = bb[gi][rhs]-lastKr[lb][i]*bb[l][rhs];
							}
						}
					}
					//else	{do nothing}							// procs after col l-1

					// send l-1 values to other procs for next iteration
					MPI_Ibcast (&bb[l-1][0], m, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_bb);
					// TODO: can be avoided?
				}


			/*
			 * update table
			 */
				// last row is computed above
				// for procs with last_row == 0, nothing to do

				// lastKr and h must be here
				MPI_Waitall(3, mpi_req_row, mpi_st_row);

				// lastKc is already there
				// no need to "MPI_Wait"

				for (i=0; i<last_row; i++)
				{
					hhh[i]  = lastKc[lb][i]*h[i];
				}

				if (mpi_rank_col_in_row == l_owner)
				{
					if (mpi_rank_col_in_row == mpi_rank_row_in_col)		// proc holding l-th col AND ON the diagonal
					{
						for (i=0; i < last_row; i++)
						{
							hh  = lastKc[lb][i]*h[i];

							// before diagonal
	#pragma ivdep
							for (j=0; j < i; j++)
							{
								Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
							}

							// on diagonal
							Tlocal[i][j] = Tlocal[i][j]*h[i];

							// after diagonal but before l-th
	#pragma ivdep
							for (j=i+1; j < l_col; j++)
							{
								Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
							}

							// on l-th
							Tlocal[i][j] = - lastKr[lb][j]*hh;

							// after l-th
	#pragma ivdep
							for (j=l_col+1; j < mycols; j++)
							{
								Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
							}
						}
					}
					else												// procs holding l-th col AND OFF the diagonal
					{
						for (i=0; i < last_row; i++)
						{
							hh  = lastKc[lb][i]*h[i];

							// before l-th
	#pragma ivdep
							for (j=0; j < l_col; j++)
							{
								Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
							}

							// on l-th
							Tlocal[i][j] = - lastKr[lb][j]*hh;

							// after l-th
	#pragma ivdep
							for (j=l_col+1; j < mycols; j++)
							{
								Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
							}
						}
					}
				}
				else if (mpi_rank_col_in_row == mpi_rank_row_in_col)	// procs ON the diagonal but NOT holding the l-th col
				{
					for (i=0; i < last_row; i++)
					{
						hh  = lastKc[lb][i]*h[i];

						// before diagonal
	#pragma ivdep
						for (j=0; j < i; j++)
						{
							Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
						}

						// on diagonal
						Tlocal[i][j] = Tlocal[i][j]*h[i];

						// after diagonal
	#pragma ivdep
						for (j=i+1; j < mycols; j++)
						{
							Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
						}
					}
				}
				else													// procs OFF the diagonal but NOT holding the l-th col
				{
					for (i=0; i < last_row; i++)
					{
						hh  = lastKc[lb][i]*h[i];

	#pragma ivdep
						for (j=0; j < mycols; j++)
						{
							Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
						}
					}
				}

			/*
			 * update local copy of global last rows and cols of K
			 */
				int current_last;
				current_last=lb;
				for (i=0;i<current_last-1;i++)
				{
					//for (j=0; j<=l-1; j++)
					for (j=0; j<mycols; j++)
					{
						lastKc[i][j]=lastKc[i][j]*h[j]   - lastKr[current_last][l-current_last+i]*hhh[j];
						lastKr[i][j]=lastKr[i][j]*h[l-current_last+i] - lastKr[current_last][j]*hhh[l-current_last+i];
					}
				}

			/*
			 * sync future last (l-1) row and col
			 */
				if (current_last==0)
				{
					if (mpi_rank==0) printf("sync\n");
					// sync last row
					if (mpi_rank_row_in_col == l_1_owner)			// procs row that holds the (l-1)-th row
					{
						for (i=0; i < bf; i++)
						{
							for (j=0; j < mycols; j++)			// copy (l-1)-th row in buffer
							{
								lastKr[i][j]=Tlocal[l_1_row-i][j];
							}
						}
					}
					MPI_Ibcast ( &lastKr[0][0], bf*mycols, MPI_DOUBLE, l_1_owner, comm_col, mpi_req_row);

					// sync last col
					if (mpi_rank_row_in_col <= l_1_owner)			// procs rows, including that one holding the (l-1)-th row
					{
						if (mpi_rank_col_in_row == l_1_owner)	// proc in the row that has (l-1)-th col
						{
							for (i=0; i < bf; i++)
							{
								for (j=0; j < last_row; j++)	// copy (l-1)-th col in buffer
								{
									lastKc[i][j]=Tlocal[j][l_1_col-i];
								}
							}
						}
						MPI_Ibcast ( &lastKc[0][0], bf*myrows, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_col);
					}
				}

		}// end of loop over levels in a block
	}// end of loop over all blocks but last one

	for (lb=bf-1; lb>0; lb--) // loop over level but last one in last block
	{
		l=block*bf+lb;
		if (mpi_rank==0) printf("level %d\n",l);
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
				/*
				if (mpi_rank_col_in_row==l_owner)
				{
					myxxstart--;
				}
				*/
				// avoid if
				myxxstart = myxxstart - (mpi_rank_col_in_row == l_owner);

				// bb[l] must be here
				//MPI_Wait( mpi_req_bb, mpi_st_bb);

				// lastKr must be here
				//MPI_Wait( mpi_req_row, mpi_st_row);

				// bb[l] and lastKr must be here
				MPI_Waitall(2, mpi_request, mpi_status);

				// update xx vector
				// l .. n-1
				for (i=myxxstart; i<myrows; i++)
				{
					gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
					for (rhs=0;rhs<m;rhs++)
					{
						xx[gi][rhs]=xx[gi][rhs]+lastKr[lb][i]*bb[l][rhs];
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
				// lastKc must be here
				//MPI_Wait( mpi_req_col, mpi_st_col);

				// lastKr must be here
				//MPI_Wait( mpi_req_row, mpi_st_row);

				MPI_Waitall(3, mpi_req_row, mpi_st_row);

#pragma ivdep
				for (i=0; i<last_row; i++)
				{
					h[i]   = 1/(1-lastKc[lb][i]*lastKr[lb][i]);
				}
			}
			MPI_Ibcast ( &h[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row, mpi_req_h);
			//TODO: async


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
							bb[gi][rhs] = bb[gi][rhs]-lastKr[lb][i]*bb[l][rhs];
						}
					}
				}
				else if ( mpi_rank_col_in_row == l_1_owner ) 		// proc holding col l-1
				{
					for (i=0; i<=l_1_col; i++)				// cols till l-1
					{
						gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
						for (rhs=0;rhs<m;rhs++)
						{
							bb[gi][rhs] = bb[gi][rhs]-lastKr[lb][i]*bb[l][rhs];
						}
					}
				}
				//else	{do nothing}							// procs after col l-1

				// send l-1 values to other procs for next iteration
				MPI_Ibcast (&bb[l-1][0], m, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_bb);
				// TODO: can be avoided?
			}


		/*
		 * update table
		 */
			// last row is computed above
			// for procs with last_row == 0, nothing to do

			// lastKr must be here
			//MPI_Wait( mpi_req_row, mpi_st_row);

			// h must be here
			//MPI_Wait( mpi_req_h, mpi_st_h);

			// lastKr and h must be here
			MPI_Waitall(3, mpi_req_row, mpi_st_row);

			// lastKc is already there
			// no need to "MPI_Wait"

			for (i=0; i<last_row; i++)
			{
				hhh[i]  = lastKc[lb][i]*h[i];
			}

			if (mpi_rank_col_in_row == l_owner)
			{
				if (mpi_rank_col_in_row == mpi_rank_row_in_col)		// proc holding l-th col AND ON the diagonal
				{
					for (i=0; i < last_row; i++)
					{
						hh  = lastKc[lb][i]*h[i];

						// before diagonal
#pragma ivdep
						for (j=0; j < i; j++)
						{
							Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
						}

						// on diagonal
						Tlocal[i][j] = Tlocal[i][j]*h[i];

						// after diagonal but before l-th
#pragma ivdep
						for (j=i+1; j < l_col; j++)
						{
							Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
						}

						// on l-th
						Tlocal[i][j] = - lastKr[lb][j]*hh;

						// after l-th
#pragma ivdep
						for (j=l_col+1; j < mycols; j++)
						{
							Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
						}
					}
				}
				else												// procs holding l-th col AND OFF the diagonal
				{
					for (i=0; i < last_row; i++)
					{
						hh  = lastKc[lb][i]*h[i];

						// before l-th
#pragma ivdep
						for (j=0; j < l_col; j++)
						{
							Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
						}

						// on l-th
						Tlocal[i][j] = - lastKr[lb][j]*hh;

						// after l-th
#pragma ivdep
						for (j=l_col+1; j < mycols; j++)
						{
							Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
						}
					}
				}
			}
			else if (mpi_rank_col_in_row == mpi_rank_row_in_col)	// procs ON the diagonal but NOT holding the l-th col
			{
				for (i=0; i < last_row; i++)
				{
					hh  = lastKc[lb][i]*h[i];

					// before diagonal
#pragma ivdep
					for (j=0; j < i; j++)
					{
						Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
					}

					// on diagonal
					Tlocal[i][j] = Tlocal[i][j]*h[i];

					// after diagonal
#pragma ivdep
					for (j=i+1; j < mycols; j++)
					{
						Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
					}
				}
			}
			else													// procs OFF the diagonal but NOT holding the l-th col
			{
				for (i=0; i < last_row; i++)
				{
					hh  = lastKc[lb][i]*h[i];

#pragma ivdep
					for (j=0; j < mycols; j++)
					{
						Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[lb][j]*hh;
					}
				}
			}


			/*
			 * update local copy of global last rows and cols of K
			 */
				int current_last;
				current_last=lb;
				for (i=0;i<current_last-1;i++)
				{
					//for (j=0; j<=l-1; j++)
					for (j=0; j<mycols; j++)
					{
						lastKc[i][j]=lastKc[i][j]*h[j]   - lastKr[current_last][l-current_last+i]*hhh[j];
						lastKr[i][j]=lastKr[i][j]*h[l-current_last+i] - lastKr[current_last][j]*hhh[l-current_last+i];
					}
				}

			/*
			 * sync future last (l-1) row and col
			 */
				if (current_last==0)
				{
					if (mpi_rank==0) printf("sync\n");
					// sync last row
					if (mpi_rank_row_in_col == l_1_owner)			// procs row that holds the (l-1)-th row
					{
						for (i=0; i < bf; i++)
						{
							for (j=0; j < mycols; j++)			// copy (l-1)-th row in buffer
							{
								lastKr[i][j]=Tlocal[l_1_row-i][j];
							}
						}
					}
					MPI_Ibcast ( &lastKr[0][0], bf*mycols, MPI_DOUBLE, l_1_owner, comm_col, mpi_req_row);

					// sync last col
					if (mpi_rank_row_in_col <= l_1_owner)			// procs rows, including that one holding the (l-1)-th row
					{
						if (mpi_rank_col_in_row == l_1_owner)	// proc in the row that has (l-1)-th col
						{
							for (i=0; i < bf; i++)
							{
								for (j=0; j < last_row; j++)	// copy (l-1)-th col in buffer
								{
									lastKc[i][j]=Tlocal[j][l_1_col-i];
								}
							}
						}
						MPI_Ibcast ( &lastKc[0][0], bf*myrows, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_col);
					}
				}

	}// end of loop over levels in the last block

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
	//MPI_Barrier(comm);

	// cleanup
	NULLFREE(lastKc);
	NULLFREE(lastKr);

	DeallocateMatrix2D(lastK,2*bf,CONTIGUOUS);
	DeallocateVector(h);
	//DeallocateVector(hh);
	DeallocateMatrix2D(Tlocal,myrows,CONTIGUOUS);

	result.total_end_time = time(NULL);

	return result;
}
