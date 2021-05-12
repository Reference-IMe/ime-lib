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

test_output pbDGESV_CO_dev ( double** A, double** bb, double** xx, test_input input, parallel_env env, int num_of_failing_ranks, int* failing_rank_list, int failing_level, int recovery_enabled )
{
	/*
	 * nb	NOT USED: blocking factor: number of adjacent column (block width)
	 * n	size (number of columns) of the square matrix A
	 * m	number of right-hand-sides number of columns) in bb
	 *
	 */

	/*
	 * rename some variables for better code readability
	 */
	int n			=	input.n;
	int nrhs		=	input.nrhs;
	int calc_procs	=	input.calc_procs;
	int spare_procs	=	input.spare_procs;
	int mpi_rank	=	env.mpi_rank;

	/*
	 * define and init main variables
	 */

	test_output result;

	result.total_start_time = time ( NULL );

	MPI_Comm comm_calc;
	MPI_Comm comm_row;
	MPI_Comm comm_row_calc;
	MPI_Comm comm_col;

	MPI_Comm comm_row_checksum;

	int calc_proc_rows;
	int calc_proc_cols;
	int spare_proc_rows;
	int spare_proc_cols;

	// calc. processes are in a ( square ) grid rows x cols
	calc_proc_rows = sqrt ( calc_procs );
	calc_proc_cols = calc_proc_rows;

	// spare processes are in a grid with the same number of rows as in calc. processes
	spare_proc_rows = calc_proc_rows;
	spare_proc_cols = spare_procs / spare_proc_rows;

	int mpi_row;
	int mpi_col;
	// separate communicators for calc. and spare processes
	// and assign row and col index
	// TODO: use mpi topologies?
	if ( likely ( mpi_rank < calc_procs ) )
	{
		MPI_Comm_split ( MPI_COMM_WORLD, 0, mpi_rank, &comm_calc );		// communicator for calc only nodes

		mpi_row = mpi_rank / calc_proc_cols;							// put nodes on grid
		mpi_col = mpi_rank % calc_proc_cols;
	}
	else
	{
		MPI_Comm_split ( MPI_COMM_WORLD, 1, mpi_rank, &comm_calc );		// communicator for spare only nodes

		mpi_row = ( mpi_rank-calc_procs ) / spare_proc_cols;
		mpi_col = calc_proc_cols + ( ( mpi_rank-calc_procs ) % spare_proc_cols );
	}

	MPI_Comm_split ( comm_calc,			mpi_row, mpi_rank, &comm_row_calc );	// communicator for a row of calc ( or spare ) only nodes
	MPI_Comm_split ( MPI_COMM_WORLD,	mpi_row, mpi_rank, &comm_row ); 		// communicator for a row
	MPI_Comm_split ( MPI_COMM_WORLD,	mpi_col, mpi_rank, &comm_col );		// communicator for a col

	int mpi_rank_col_in_row;
	int mpi_rank_row_in_col;

	MPI_Comm_rank ( comm_row, &mpi_rank_col_in_row );		// get current process id in row
	MPI_Comm_rank ( comm_col, &mpi_rank_row_in_col );		// get current process id in col

	// request handler for async. communications
	MPI_Status mpi_status[4];
	MPI_Request mpi_request[4];
				mpi_request[0]	= MPI_REQUEST_NULL;
				mpi_request[1]	= MPI_REQUEST_NULL;
				mpi_request[2]	= MPI_REQUEST_NULL;
				mpi_request[3]	= MPI_REQUEST_NULL;
	//aliases
	MPI_Request* mpi_req_bb		= &mpi_request[0]; // req. for bb broadcast
	MPI_Request* mpi_req_row	= &mpi_request[1]; // req. for row broadcast
	MPI_Request* mpi_req_h		= &mpi_request[2]; // req. for h broadcast
	MPI_Request* mpi_req_col	= &mpi_request[3]; // req. for col broadcast

	MPI_Status* mpi_st_bb		= &mpi_status[0];
	MPI_Status* mpi_st_row		= &mpi_status[1];
	//MPI_Status* mpi_st_h		= &mpi_status[2]; // never referenced explicitly
	//MPI_Status* mpi_st_col	= &mpi_status[3]; // never referenced explicitly

	int i, j, l, gi;						// general indexes
	int rhs;								// r.h.s. index
	int myrows = n / calc_proc_rows;		// num of matrix rows per process
	int mycols = n / calc_proc_cols;		// num of matrix cols per process
	int myxxrows = mycols;					// num of chunks for better code readability

	/*
	 * check grid
	 */
	/*
	for ( i=0; i < calc_procs+spare_procs; i++ )
	{
		MPI_Barrier ( MPI_COMM_WORLD );
		if ( mpi_rank==i )
		{
			printf ( "%d@ ( %d, %d ) = ( %d, %d ) \n", mpi_rank, mpi_row, mpi_col, mpi_rank_row_in_col, mpi_rank_col_in_row );
			fflush ( stdout );
		}
	}
	*/

	/*
	 * local storage for a part of the input matrix ( continuous columns, not interleaved )
	 */
	double** Tlocal;
			 Tlocal = AllocateMatrix2D ( myrows, mycols, CONTIGUOUS );

	// last row and col of K
	double** lastK;
			 lastK = AllocateMatrix2D ( 2, mycols, CONTIGUOUS );
	// aliases
	double* lastKr = &lastK[0][0];
	double* lastKc = &lastK[1][0];

	// helper vectors
	double* h;
			h = AllocateVector ( myrows );

	/*
	 * init inhibition table
	 */
	//TODO: pass also the weights
	pbDGEIT_CX_bf1_ft ( A, Tlocal, lastKr, lastKc, n, calc_procs,
			spare_proc_cols,
			comm_calc, mpi_rank,
			comm_row, mpi_rank_col_in_row,
			comm_col, mpi_rank_row_in_col,
			comm_row_calc,
			mpi_status,
			mpi_request
			 );											// init inhibition table

	//the checksum weights
	double** w;
	w = AllocateMatrix2D ( spare_proc_cols, spare_proc_rows, CONTIGUOUS ); //TODO: transpose to gain some cache hit in loops below
	RandomMatrix2D ( w, spare_proc_cols, spare_proc_rows, 0 );


	if ( mpi_rank_row_in_col == 0 && mpi_rank_col_in_row < calc_proc_cols ) 			// first row of calc procs
	{
		MPI_Ibcast ( &bb[0][0], n*nrhs, MPI_DOUBLE, 0, comm_row_calc, mpi_req_bb );	// send all r.h.s
		DGEZR ( xx, n, nrhs );															// init ( zero ) solution vectors
	}

		/*
		 * check initialization
		 */
		/*
		MPI_Waitall ( 4, mpi_request, mpi_status );

		for ( i=0; i < calc_procs+spare_procs; i++ )
		{
			MPI_Barrier ( MPI_COMM_WORLD );
			if ( mpi_rank == i )
			{
				printf ( "\n%d@ ( %d, %d ) \n", n, mpi_rank_row_in_col, mpi_rank_col_in_row );
				PrintMatrix2D ( Tlocal, myrows, mycols );
				printf ( "\n" );
				PrintMatrix2D ( lastK, 2, mycols );

				fflush ( stdout );
			}
		}
		*/


	/*
	 * calc inhibition sequence
	 */
		result.core_start_time = time ( NULL );

		// general bounds for the loops over the columns
		// they differ on processes and change along the main loop over the levels
		int myxxstart = mycols;		// beginning column position for updating the solution (begins from right)
		int last_row;				// (local) last row to process
		int l_1_owner;				// proc rank hosting the (l-1) -th row (or col)
		int l_1_row;				// local index of the (l-1) -th row
		int l_1_col;				// local index of the (l-1) -th col
		int l_owner;				// proc rank hosting the l-th row (or col)
		int l_row;					// local index of the l-th row
		int l_col;					// local index of the l-th col

		int fr;						// index for list of faulty ranks
		int faulted = 0;			// flag for faulty (1) or healthy (0) rank

		int mpi_row_with_faults;
		mpi_row_with_faults = failing_rank_list[0] / calc_proc_cols;



		// for a better optimization, treat calc procs differently from spare procs
		if ( mpi_rank_col_in_row < calc_proc_cols )											// calc. procs
		{
			if ( mpi_rank_row_in_col == 0 ) 					// first row of calc procs
			{
				// all levels but last one (l=0)
				for ( l=n-1; l > 0; l-- )
				{
					if ( unlikely ( l == failing_level ) )		// inject failure if failing level has been reached
																// (assuming only calc procs can be faulty)
																// and then recover
					{
						if ( mpi_rank_row_in_col == mpi_row_with_faults )
						{
							// fault injection
							for ( fr=0; fr < num_of_failing_ranks; fr++ )
							{
								if ( unlikely ( mpi_rank == failing_rank_list[fr] ) )
								{
									printf ( "## IMe: rank %d faulty at level %d\n", mpi_rank, l );
									SetMatrix2D ( -99, Tlocal, myrows, mycols );
									faulted=1;
								}
							}
							/*
							 * check fault injection
							 */
							/*
							for ( fr=0; fr < num_of_failing_ranks; fr++ )
							{
								if ( mpi_rank == failing_rank_list[fr] )
								{
									printf ( " * before fault:\n" );
									PrintMatrix2D ( Tlocal, myrows, mycols );

									printf ( " * after fault:\n" );
									PrintMatrix2D ( Tlocal, myrows, mycols );
								}
							}
							*/

							if (recovery_enabled)
							{
								double** tmpTlocal;
										 tmpTlocal = AllocateMatrix2D ( myrows, mycols, CONTIGUOUS );

								/*
								 *  update sums
								 *
								 *  for every faulty proc in the row a checksum proc has to provide recovery data
								 *  the basic recovery data is the weighted sum of the contribution of the faulty procs
								 *  this can be calculated from the full weighted sum diminished by the contribution of the non-faulty procs
								 */
								for ( fr=0; fr < num_of_failing_ranks; fr++ )
								{
									if ( !faulted ) // non-faulty procs prepare contributions to send to the checksum node
									{
										// TODO: last row to recover ?
										for ( i=0; i < myrows; i++ )
										{
											#pragma ivdep
											for ( j=0; j < mycols; j++ )
											{
												tmpTlocal[i][j] = -Tlocal[i][j] * w[fr][mpi_rank_col_in_row];
											}
										}
										if ( unlikely ( mpi_rank_col_in_row == mpi_rank_row_in_col ) )
										{
											for ( j=0; j < mycols; j++ )
											{
												#pragma ivdep
												tmpTlocal[j][j] = tmpTlocal[j][j] - w[fr][mpi_rank_col_in_row];
											}
										}
										MPI_Comm_split ( comm_row, 1, mpi_rank, &comm_row_checksum );
										MPI_Reduce ( &tmpTlocal[0][0], NULL, myrows*mycols, MPI_DOUBLE, MPI_SUM, calc_proc_cols - spare_proc_cols, comm_row_checksum );
									}
									else // faulty procs do nothing
									{
										MPI_Comm_split ( comm_row, 0, mpi_rank, &comm_row_checksum );
										// does not participate in reduction
									}
									MPI_Comm_free ( &comm_row_checksum );
								}

								/*
								 * without loss of generality, faulty procs are revived to be recovered
								 */
								for ( fr=0; fr < num_of_failing_ranks; fr++ )
								{
									if ( unlikely ( mpi_rank == failing_rank_list[fr] ) ) // revived procs prepare to receive sums
									{
										for ( i=0; i < myrows; i++ )
										{
											#pragma ivdep
											for ( j=0; j < mycols; j++ )
											{
												tmpTlocal[i][j] = 0;
											}
										}
										MPI_Comm_split ( comm_row, 1, mpi_rank, &comm_row_checksum );
										MPI_Reduce ( &tmpTlocal[0][0], &Tlocal[0][0], myrows*mycols, MPI_DOUBLE, MPI_SUM, 0, comm_row_checksum );

										// because diagonal elements on diagonal procs are added with one before checksumming
										// once received the sums, those elements have to be decremented to reflect the actual values
										if ( unlikely ( mpi_rank_col_in_row == mpi_rank_row_in_col ) )
										{
											for ( j=0; j < mycols; j++ )
											{
												#pragma ivdep
												Tlocal[j][j] = Tlocal[j][j] - 1;
											}
										}
										printf ( "## IMe: rank %d recovered at level %d\n", mpi_rank, l );
									}
									else
									{
										MPI_Comm_split ( comm_row, 0, mpi_rank, &comm_row_checksum );
										// does not participate in reduction
									}
									MPI_Comm_free ( &comm_row_checksum );
								}
								/*
								 * check recovery
								*/
								/*
								for ( fr=0; fr < num_of_failing_ranks; fr++ )
								{
									MPI_Barrier ( comm_row_calc );
									if ( mpi_rank == failing_rank_list[fr] )
									{
										printf ( "\n * rank %d recovered *\n", mpi_rank, l );

										printf ( " * with:\n" );
										PrintMatrix2D ( Tlocal, myrows, mycols );
									}
								}
								*/

								DeallocateMatrix2D ( tmpTlocal, myrows, CONTIGUOUS );
							}
						}
					}// failure management

					l_owner = PVMAP ( l, myrows );
					l_row = PVLOCAL ( l, myrows );
					l_col = l_row;

					l_1_owner = PVMAP ( l-1, mycols );
					l_1_col = PVLOCAL ( l-1, mycols );
					l_1_row = l_1_col;

					// last_row = myrows | l_row | 0
					last_row = ( mpi_rank_row_in_col < l_owner ) * myrows + ( mpi_rank_row_in_col == l_owner ) * l_row;

					/*
					 * update solutions
					 */
					MPI_Waitall ( 2, mpi_request, mpi_status );	// wait for lastKr and bb
					pbDGEUX_CO ( 	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, nrhs,
									l, l_owner, l_1_owner, l_row, l_col,
									&myxxstart,
									lastKr, xx, bb );

					/*
					 * update helpers
					 */
					if ( mpi_rank_row_in_col == mpi_rank_col_in_row )
					{
						MPI_Waitall ( 3, mpi_req_row, mpi_st_row ); // wait for lastKr and lastKc
						pbDGEUH_CO ( 	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, nrhs,
										l, l_owner, l_1_owner, l_row, l_col,
										last_row,
										lastKr, lastKc, h );
					}
					MPI_Ibcast ( &h[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row, mpi_req_h );

					/*
					 * update auxiliary causes
					 */
					pbDGEUB_CO ( 	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, nrhs,
									l, l_owner, l_1_owner, l_row, l_col,
									l_1_col,
									lastKr, bb );
					MPI_Ibcast ( &bb[l-1][0], nrhs, MPI_DOUBLE, l_1_owner, comm_row_calc, mpi_req_bb );

					/*
					 * update table
					 */
					MPI_Waitall ( 3, mpi_req_row, mpi_st_row ); // wait for lastKr, lastKc and h
					pbDGEUT_CO ( 	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, nrhs,
									l, l_owner, l_1_owner, l_row, l_col,
									last_row,
									lastKr, lastKc, h, Tlocal );

					/*
					 * sync future last (l-1) row and col
					 */
						// sync last row
						if ( mpi_rank_row_in_col == l_1_owner ) 			// procs row that holds the (l-1)-th row
						{
							/*
							#pragma ivdep
							for ( i=0; i < mycols; i++ ) 			// copy (l-1)-th row in buffer
							{
								lastKr[i]=Tlocal[l_1_row][i];
							}
							*/
							lastKr=Tlocal[l_1_row];
						}
						MPI_Ibcast ( &lastKr[0], mycols, MPI_DOUBLE, l_1_owner, comm_col, mpi_req_row );

						// sync last col
						if ( mpi_rank_row_in_col <= l_1_owner ) 			// procs rows, including that one holding the ( l-1 ) -th row
						{
							if ( mpi_rank_col_in_row == l_1_owner ) 	// proc in the row that has ( l-1 ) -th col
							{
								#pragma ivdep
								for ( i=0; i < last_row; i++ ) 	// copy ( l-1 ) -th col in buffer
								{
									lastKc[i]=Tlocal[i][l_1_col];
								}
							}
							MPI_Ibcast ( &lastKc[0], last_row, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_col );
						}
				}// end of loop over levels
			}
			else
			{
				// all levels but last one (l=0)
				for ( l=n-1; l > 0; l-- )
				{
					if ( unlikely ( l == failing_level ) )
					{
						if ( mpi_rank_row_in_col == mpi_row_with_faults ) // only spare procs in a row with a fault have things to do
						{
							// fault injection
							for ( fr=0; fr < num_of_failing_ranks; fr++ )
							{
								if ( unlikely ( mpi_rank == failing_rank_list[fr] ) )
								{
									printf ( "## IMe: rank %d faulty at level %d\n", mpi_rank, l );
									SetMatrix2D ( -99, Tlocal, myrows, mycols );
									faulted=1;
								}
							}
							/*
							 * check fault injection
							 */
							/*
							for ( fr=0; fr < num_of_failing_ranks; fr++ )
							{
								if ( mpi_rank == failing_rank_list[fr] )
								{
									printf ( " * before fault:\n" );
									PrintMatrix2D ( Tlocal, myrows, mycols );

									printf ( " * after fault:\n" );
									PrintMatrix2D ( Tlocal, myrows, mycols );
								}
							}
							*/

							if (recovery_enabled)
							{
								double** tmpTlocal;
										 tmpTlocal = AllocateMatrix2D ( myrows, mycols, CONTIGUOUS );

								/*
								 *  update sums
								 *
								 *  for every faulty proc in the row a checksum proc has to provide recovery data
								 *  the basic recovery data is the weighted sum of the contribution of the faulty procs
								 *  this can be calculated from the full weighted sum diminished by the contribution of the non-faulty procs
								 */
								for ( fr=0; fr < num_of_failing_ranks; fr++ )
								{
									if ( !faulted ) // non-faulty procs prepare contributions to send to the checksum node
									{
										for ( i=0; i < myrows; i++ )
										{
											#pragma ivdep
											for ( j=0; j < mycols; j++ )
											{
												tmpTlocal[i][j] = -Tlocal[i][j] * w[fr][mpi_rank_col_in_row];
											}
										}
										if ( unlikely ( mpi_rank_col_in_row == mpi_rank_row_in_col ) )
										{
											for ( j=0; j < mycols; j++ )
											{
												#pragma ivdep
												tmpTlocal[j][j] = tmpTlocal[j][j] - w[fr][mpi_rank_col_in_row];
											}
										}
										MPI_Comm_split ( comm_row, 1, mpi_rank, &comm_row_checksum );
										MPI_Reduce ( &tmpTlocal[0][0], NULL, myrows*mycols, MPI_DOUBLE, MPI_SUM, calc_proc_cols - spare_proc_cols, comm_row_checksum );
									}
									else // faulty procs do nothing
									{
										MPI_Comm_split ( comm_row, 0, mpi_rank, &comm_row_checksum );
										// does not participate in reduction
									}
									MPI_Comm_free ( &comm_row_checksum );
								}

								/*
								 * without loss of generality, faulty procs are revived to be recovered
								 */
								for ( fr=0; fr < num_of_failing_ranks; fr++ )
								{
									if ( unlikely ( mpi_rank == failing_rank_list[fr] ) ) // revived procs prepare to receive sums
									{
										for ( i=0; i < myrows; i++ )
										{
											#pragma ivdep
											for ( j=0; j < mycols; j++ )
											{
												tmpTlocal[i][j] = 0;
											}
										}
										MPI_Comm_split ( comm_row, 1, mpi_rank, &comm_row_checksum );
										MPI_Reduce ( &tmpTlocal[0][0], &Tlocal[0][0], myrows*mycols, MPI_DOUBLE, MPI_SUM, 0, comm_row_checksum );

										// because diagonal elements on diagonal procs are added with one before checksumming
										// once received the sums, those elements have to be decremented to reflect the actual values
										if ( unlikely ( mpi_rank_col_in_row == mpi_rank_row_in_col ) )
										{
											for ( j=0; j < mycols; j++ )
											{
												#pragma ivdep
												Tlocal[j][j] = Tlocal[j][j] - 1;
											}
										}
										printf ( "## IMe: rank %d recovered at level %d\n", mpi_rank, l );
									}
									else
									{
										MPI_Comm_split ( comm_row, 0, mpi_rank, &comm_row_checksum );
										// does not participate in reduction
									}
									MPI_Comm_free ( &comm_row_checksum );
								}
								/*
								 * check recovery
								*/
								/*
								for ( fr=0; fr < num_of_failing_ranks; fr++ )
								{
									MPI_Barrier ( comm_row_calc );
									if ( mpi_rank == failing_rank_list[fr] )
									{
										printf ( "\n * rank %d recovered *\n", mpi_rank, l );

										printf ( " * with:\n" );
										PrintMatrix2D ( Tlocal, myrows, mycols );
									}
								}
								*/

								DeallocateMatrix2D ( tmpTlocal, myrows, CONTIGUOUS );
							}
						}
					}// failure management

					l_owner = PVMAP ( l, myrows );
					l_row = PVLOCAL ( l, myrows );
					l_col = l_row;

					l_1_owner = PVMAP ( l-1, mycols );
					l_1_col = PVLOCAL ( l-1, mycols );
					l_1_row = l_1_col;

					// last_row = myrows | l_row | 0
					last_row = ( mpi_rank_row_in_col < l_owner ) * myrows + ( mpi_rank_row_in_col == l_owner ) * l_row;

					/*
					 * update helpers
					 */
					if ( mpi_rank_row_in_col == mpi_rank_col_in_row )
					{
						MPI_Waitall ( 3, mpi_req_row, mpi_st_row ); // wait for lastKr and lastKc
						pbDGEUH_CO ( 	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, nrhs,
										l, l_owner, l_1_owner, l_row, l_col,
										last_row,
										lastKr, lastKc, h );
					}
					MPI_Ibcast ( &h[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row, mpi_req_h );

					/*
					 * update table
					 */
					MPI_Waitall ( 3, mpi_req_row, mpi_st_row ); // wait for lastKr, lastKc and h
					pbDGEUT_CO ( 	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, nrhs,
									l, l_owner, l_1_owner, l_row, l_col,
									last_row,
									lastKr, lastKc, h, Tlocal );

					/*
					 * sync future last (l-1) row and col
					 */
						// sync last row
						if ( mpi_rank_row_in_col == l_1_owner ) 			// procs row that holds the (l-1) -th row
						{
							/*
							#pragma ivdep
							for ( i=0; i < mycols; i++ ) 			// copy (l-1) -th row in buffer
							{
								lastKr[i]=Tlocal[l_1_row][i];
							}
							*/
							lastKr=Tlocal[l_1_row];
						}
						MPI_Ibcast ( &lastKr[0], mycols, MPI_DOUBLE, l_1_owner, comm_col, mpi_req_row );

						// sync last col
						if ( mpi_rank_row_in_col <= l_1_owner ) 			// procs rows, including that one holding the ( l-1 ) -th row
						{
							if ( mpi_rank_col_in_row == l_1_owner ) 	// proc in the row that has ( l-1 ) -th col
							{
								#pragma ivdep
								for ( i=0; i < last_row; i++ ) 	// copy ( l-1 ) -th col in buffer
								{
									lastKc[i] = Tlocal[i][l_1_col];
								}
							}
							MPI_Ibcast ( &lastKc[0], last_row, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_col );
						}
				}// end of loop over levels
			}
		}
		// for a better optimization, treat calc procs differently from spare procs
		else																				// spare procs
		{
			// all levels but last one (l=0)
			for ( l=n-1; l > 0; l-- )
			{
				if ( unlikely ( l == failing_level ) )			// inject failure
																// fault is injected in calc procs (above)
				{
					if ( mpi_rank_row_in_col == mpi_row_with_faults ) // only spare procs in a row with a fault have things to do
					{
						if (recovery_enabled)
						{
							double** tmpTlocal;
									 tmpTlocal = AllocateMatrix2D ( myrows, mycols, CONTIGUOUS );

							double** wfaulty;
									 wfaulty = AllocateMatrix2D ( spare_proc_cols, spare_proc_cols, CONTIGUOUS );

							// weights matrix of the faulty ones
							// TODO: generalize column extraction for non consecutive ranks
							for ( i=0; i < num_of_failing_ranks; i++ )
							{
								for ( j=0; j < num_of_failing_ranks; j++ )
								{
									wfaulty[i][j] = w[i][failing_rank_list[0] - mpi_row_with_faults * calc_proc_cols + j];
								}
							}

							if ( mpi_rank_col_in_row == calc_proc_cols ) // first spare proc in row
							{
								printf ( "\nweights:\n" );
								PrintMatrix2D ( w, spare_proc_cols, spare_proc_rows );
								printf ( "faulty weights:\n" );
								PrintMatrix2D ( wfaulty, spare_proc_cols, num_of_failing_ranks );
							}

							int* ipiv = malloc ( num_of_failing_ranks*sizeof ( int ) );
							int lwork = num_of_failing_ranks*num_of_failing_ranks;
							double* work = malloc ( lwork*sizeof ( double ) );
							int info;
							dgetrf_ ( &num_of_failing_ranks, &num_of_failing_ranks, &wfaulty[0][0], &num_of_failing_ranks, ipiv, &info );
							dgetri_ ( &num_of_failing_ranks, &wfaulty[0][0], &num_of_failing_ranks, ipiv, work, &lwork, &info );

							if ( mpi_rank_col_in_row == calc_proc_cols )
							{
								printf ( "recovery weights:\n" );
								PrintMatrix2D ( wfaulty, spare_proc_cols, num_of_failing_ranks );
							}

							/*
							 * update sums
							 *
							 * every checksum node have to diminish its sum by the weighted contribution of the non-faulty procs
							 */
							for ( fr=0; fr < spare_proc_cols; fr++ )
							{
								if ( unlikely ( mpi_rank_col_in_row == ( calc_proc_cols+fr ) ) ) // receive the contributions
								{
									MPI_Comm_split ( comm_row, 1, mpi_rank, &comm_row_checksum );
									MPI_Reduce ( &Tlocal[0][0], &tmpTlocal[0][0], myrows*mycols, MPI_DOUBLE, MPI_SUM, calc_proc_cols - spare_proc_cols, comm_row_checksum );
								}
								else
								{
									MPI_Comm_split ( comm_row, 0, mpi_rank, &comm_row_checksum );
									// does not participate in reduction
								}
								MPI_Comm_free ( &comm_row_checksum );
							}

							/*
							 * send sums
							 */
							for ( fr=0; fr < num_of_failing_ranks; fr++ )
							{
								for ( i=0; i < myrows; i++ )
								{
									#pragma ivdep
									for ( j=0; j < mycols; j++ )
									{
										Tlocal[i][j] = tmpTlocal[i][j] * wfaulty[fr][mpi_rank_col_in_row-calc_proc_cols];
									}
								}
								MPI_Comm_split ( comm_row, 1, mpi_rank, &comm_row_checksum );
								MPI_Reduce ( &Tlocal[0][0], NULL, myrows*mycols, MPI_DOUBLE, MPI_SUM, 0, comm_row_checksum );

								MPI_Comm_free ( &comm_row_checksum );
							}

							NULLFREE(work);
							NULLFREE(ipiv);
							DeallocateMatrix2D ( wfaulty, spare_proc_cols, CONTIGUOUS );
							DeallocateMatrix2D ( tmpTlocal, myrows, CONTIGUOUS );
						}
					}
				}

				l_owner = PVMAP ( l, myrows );
				l_row = PVLOCAL ( l, myrows );
				l_col = l_row;

				l_1_owner = PVMAP ( l-1, mycols );
				l_1_col = PVLOCAL ( l-1, mycols );
				l_1_row = l_1_col;

				// last_row = myrows | l_row | 0
				last_row = ( mpi_rank_row_in_col < l_owner ) * myrows + ( mpi_rank_row_in_col == l_owner ) * l_row;

				/*
				 * update helpers
				 */
				MPI_Ibcast ( &h[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row, mpi_req_h );

				/*
				 * update table
				 */
				MPI_Waitall ( 3, mpi_req_row, mpi_st_row ); // wait for lastKr, lastKc and h
				pbDGEUT_CO ( 	mpi_rank_row_in_col, mpi_rank_col_in_row, myrows, mycols, nrhs,
								l, l_owner, l_1_owner, l_row, l_col,
								last_row,
								lastKr, lastKc, h, Tlocal );

				/*
				 * sync future last (l-1) row and col
				 */
					// sync last row
					if ( mpi_rank_row_in_col == l_1_owner ) 			// procs row that holds the (l-1) -th row
					{
						/*
						#pragma ivdep
						for ( i=0; i < mycols; i++ ) 			// copy (l-1) -th row in buffer
						{
							lastKr[i]=Tlocal[l_1_row][i];
						}
						*/
						lastKr=Tlocal[l_1_row];
					}
					MPI_Ibcast ( &lastKr[0], mycols, MPI_DOUBLE, l_1_owner, comm_col, mpi_req_row );

					// sync last col
					if ( mpi_rank_row_in_col <= l_1_owner ) 			// procs rows, including that one holding the (l-1) -th row
					{
						if ( mpi_rank_col_in_row == l_1_owner ) 	// proc in the row that has (l-1) -th col
						{
							#pragma ivdep
							for ( i=0; i < last_row; i++ ) 	// copy (l-1) -th col in buffer
							{
								lastKc[i] = Tlocal[i][l_1_col];
							}
						}
						MPI_Ibcast ( &lastKc[0], last_row, MPI_DOUBLE, l_1_owner, comm_row, mpi_req_col );
					}
			}// end of loop over levels
		}

		/*
		 * update last level of the table
		 */
			// last level (l=0)
			if ( mpi_rank_row_in_col == 0 && mpi_rank_col_in_row < calc_proc_cols ) 								// first row of calc procs
			{
				// bb[0] must be here
				MPI_Wait ( mpi_req_bb, mpi_st_bb );

				#pragma ivdep
				for ( i=0; i < myxxrows; i++ )
				{
					gi=PVGLOBAL ( i, mycols, mpi_rank_col_in_row );
					for ( rhs=0; rhs < nrhs; rhs++ )
					{
						xx[gi][rhs]=xx[gi][rhs]+Tlocal[0][i]*bb[0][rhs];
					}
				}

				result.core_end_time = time ( NULL );

				// TODO: add checking on exit code
				result.exit_code = 0;

				// collect solution
				// MPI_IN_PLACE required for MPICH based versions
				if ( mpi_rank_col_in_row==0 )
				{
					MPI_Gather ( MPI_IN_PLACE, nrhs*myxxrows, MPI_DOUBLE, &xx[0][0], nrhs*myxxrows, MPI_DOUBLE, 0, comm_row_calc );
				}
				else
				{
					MPI_Gather ( &xx[mpi_rank_col_in_row*myxxrows][0], nrhs*myxxrows, MPI_DOUBLE, &xx[0][0], nrhs*myxxrows, MPI_DOUBLE, 0, comm_row_calc );
				}
			}
			else
			{
				result.core_end_time = time ( NULL );

				// TODO: add checking on exit code
				result.exit_code = 0;
			}

			/*
			 * check table result
			 */
			/*
			MPI_Barrier ( MPI_COMM_WORLD );
			if ( mpi_rank_row_in_col == 0 ) 								// first row of calc procs
			{
				for ( i=0; i < calc_proc_cols+spare_proc_cols; i++ )
				{
					MPI_Barrier ( comm_row );
					if ( mpi_rank_col_in_row==i )
					{
						printf ( "\nl=0 @ ( %d, %d ) :\n", mpi_rank_row_in_col, mpi_rank_col_in_row );
						PrintMatrix2D ( Tlocal, myrows, mycols );
						printf ( "\n" );

						fflush ( stdout );
					}
					MPI_Barrier ( comm_row );
				}
			}
			*/


	// wait to complete before cleanup and return
	MPI_Waitall ( 4, mpi_request, mpi_status );

	// cleanup
	DeallocateMatrix2D ( w, spare_proc_cols, CONTIGUOUS );
	DeallocateMatrix2D ( lastK, 2, CONTIGUOUS );
	DeallocateVector   ( h );
	DeallocateMatrix2D ( Tlocal, myrows, CONTIGUOUS );

	result.total_end_time = time ( NULL );

	return result;
}
