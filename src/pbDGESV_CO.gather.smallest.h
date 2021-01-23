#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "helpers/macros.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "DGEZR.h"
#include "pbDGEIT_CX.h"


test_output pbDGESV_CO_g_smallest(int nb, int n, double** A, int m, double** bb, double** xx, MPI_Comm comm)
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

	//MPI_Status  mpi_status;
	//MPI_Request mpi_request = MPI_REQUEST_NULL;

	int i,j,l;						// general indexes
    int rhs;						// r.h.s. index
    int myrows   = n/cprocrows;		// num of rows per process
    int mycols   = n/cproccols;		// num of cols per process
    int myxxrows = mycols;			// num of chunks for better code readability
    int bf           = 1;			// blocking factor
	int current_last = bf-1;		// index for the current last row or col of K in buffer
    //TODO: use nb as blocking factor

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Tlocal;
			 Tlocal=AllocateMatrix2D(myrows, mycols, CONTIGUOUS);
	// aliases for better code readability
	/*
    // X part
    double** Xlocal;
    		 Xlocal=Tlocal;
    // K part
	double** Klocal;
			 Klocal=Tlocal;
	*/

	// last rows and cols of K
	double** lastK;
			 lastK=AllocateMatrix2D(2*nb, mycols, CONTIGUOUS);	// last rows [0 - (bf-1)] and cols [ bf - (2bf -1)] of K
	double** lastKr;
				lastKr=malloc(bf*sizeof(double*));
				for(i=0;i<bf;i++)
				{
					lastKr[i]=lastK[i];						// alias for last row
				}
	double** lastKc;
				lastKc=malloc(nb*sizeof(double*));
				for(i=0;i<bf;i++)
				{
					lastKc[i]=lastK[bf+i];					// alias for last col
				}
	// helper vectors
    double* h;
    		h=AllocateVector(myrows);
    double* hh;
			hh=AllocateVector(myrows);

	/*
	 *  init inhibition table
	 */
	DGEZR(xx, n, m);											// init (zero) solution vectors
	pbDGEIT_CX(A, Tlocal, lastK, n, bf,
			comm, mpi_rank, comm_row,
			mpi_rank_col_in_row, comm_col,
			mpi_rank_row_in_col,
			cprocs);											// init inhibition table

	if (mpi_rank_row_in_col==0) 								// first row of procs
	{
		MPI_Bcast (&bb[0][0], n*m, MPI_DOUBLE, 0, comm_row);	// send all r.h.s
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
	int gj;			// global j
	int l_owner;	// proc rank hosting the l-th row (or col)
	int l_1_owner;	// proc rank hosting the (l-1)-th row (or col)
	int l_1_col;	// local index of the (l-1)-th col

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		l_owner  = PVMAP(l, myrows);
		l_1_owner = PVMAP(l-1, mycols);
		l_1_col   = PVLOCAL(l-1, mycols);

		/*
		if (mpi_rank==0)
		{
			printf("\n[%d]\n",l);
		}
		MPI_Barrier(MPI_COMM_WORLD);

		for (i=0; i<cprocs; i++)
		{
			MPI_Barrier(MPI_COMM_WORLD);
			if (mpi_rank==i)
			{
				printf("Tlocal and lastK in %d (%d,%d):\n",mpi_rank,mpi_rank_row_in_col,mpi_rank_col_in_row);
				PrintMatrix2D(Tlocal, myrows, mycols);
				printf("\n");
				PrintMatrix2D(lastK, 2*bf, mycols);
				printf("\n");
			    fflush(stdout);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		*/

		/*
		 * update solutions
		 */
			if (mpi_rank_row_in_col==0) 					// first row of procs
			{
				if (mpi_rank_col_in_row==l_owner) 		// if a process contains the l-th cols, must skip it
				{
					myxxstart--;
				}
				//TODO: avoid if?

				// update xx vector
				// l .. n-1
				//for (i=myxxstart; i<=PVLOCAL(n-1, mycols); i++)
				for (i=myxxstart; i<myrows; i++)
				{
					gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
					for (rhs=0;rhs<m;rhs++)
					{
						xx[gi][rhs]=xx[gi][rhs]+lastKr[current_last][i]*bb[l][rhs];
					}
				}
			}


		/*
		 * update helpers
		 */
			// chunks of last row and col are symmetrical to the main diagonal
			// they are hosted on procs on the diagonal
			if ( mpi_rank_row_in_col < l_owner )					// procs before row l
			{
				last_row=myrows;								// all rows
				if (mpi_rank_row_in_col==mpi_rank_col_in_row)
				{
					for (i=0; i<last_row; i++)
					{
						h[i]   = 1/(1-lastKc[current_last][i]*lastKr[current_last][i]);
					}
				}
				MPI_Bcast ( &h[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row);

				for (i=0; i<last_row; i++)
				{
					hh[i]  = lastKc[current_last][i]*h[i];
				}
			}
			else if ( mpi_rank_row_in_col == l_owner )				// procs holding row l
			{
				last_row=PVLOCAL(l, myrows);					// rows till l-1
				if (mpi_rank_row_in_col==mpi_rank_col_in_row)
				{
					for (i=0; i<last_row; i++)
					{
						h[i]   = 1/(1-lastKc[current_last][i]*lastKr[current_last][i]);
					}
				}
				MPI_Bcast ( &h[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row);

				for (i=0; i<last_row; i++)
				{
					hh[i]  = lastKc[current_last][i]*h[i];
				}
			}
			else													// procs after row l-1
			{
				last_row=-1;									// no rows
				// {do nothing}
			}


		/*
		 * update auxiliary causes
		 */
			if (mpi_rank_row_in_col==0) 							// first row of procs
			{
				if ( mpi_rank_col_in_row < l_1_owner ) 			// procs before col l-1
				{
					for (i=0; i<mycols; i++)				// treat all cols
					{
						gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
						for (rhs=0;rhs<m;rhs++)
						{
							bb[gi][rhs] = bb[gi][rhs]-lastKr[current_last][i]*bb[l][rhs];
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
							bb[gi][rhs] = bb[gi][rhs]-lastKr[current_last][i]*bb[l][rhs];
						}
					}
				}
				//else	{do nothing}							// procs after col l-1

				// send l-1 values to other procs for next iteration
				MPI_Bcast (&bb[l-1][0], m, MPI_DOUBLE, l_1_owner, comm_row);
			}

		/*
		// print h and hh
		for (i=0; i<cprocs; i++)
		{
			MPI_Barrier(MPI_COMM_WORLD);
			if (mpi_rank==i)
			{
				printf("h and hh in %d (%d,%d):\n",mpi_rank,mpi_rank_row_in_col,mpi_rank_col_in_row);
				PrintVector(h, myrows);
				printf("\n");
				PrintVector(hh, myrows);
				printf("\n");
			    fflush(stdout);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		*/

		/*
		 * update table
		 */
			// last row is computed above
			// for procs with last_row == -1, nothing to do
			for (i=0; i < last_row; i++)
			{
				gi=PVGLOBAL(i, myrows, mpi_rank_row_in_col);
				for (j=0; j<mycols; j++)
				{
					gj=PVGLOBAL(j, mycols, mpi_rank_col_in_row);
					if (gj==gi)
					{
						Tlocal[i][j] = Tlocal[i][j]*h[i];
					}
					else if (gj==l)
					{
						Tlocal[i][j] = - lastKr[current_last][j]*hh[i];
					}
					else
					{
						Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[current_last][j]*hh[i];
					}
				}
			}


		/*
		 * sync future last (l-1) row and col
		 */
			// sync last col
			if (mpi_rank_row_in_col <= l_1_owner)			// procs rows, including that one holding the (l-1)-th row
			{
				if (mpi_rank_col_in_row == l_1_owner)	// proc in the row that has (l-1)-th col
				{
					for (i=0; i < last_row; i++)	// copy (l-1)-th col in buffer
					{
						lastKc[current_last][i]=Tlocal[i][l_1_col];
					}
				}
				MPI_Bcast ( &lastKc[current_last][0], last_row, MPI_DOUBLE, l_1_owner, comm_row);
			}

			// sync last row
			if (mpi_rank_row_in_col == l_1_owner)			// procs row that holds the (l-1)-th row
			{
				for (i=0; i < mycols; i++)			// copy (l-1)-th row in buffer
				{
					lastKr[current_last][i]=Tlocal[l_1_col][i];
				}
			}
			MPI_Bcast ( &lastKr[current_last][0], mycols, MPI_DOUBLE, l_1_owner, comm_col);

	}// end of loop over levels

	/*
	for (i=0; i<cprocs; i++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		if (mpi_rank==i)
		{
			printf("[0]\nTlocal and lastK in %d (%d,%d):\n",mpi_rank,mpi_rank_row_in_col,mpi_rank_col_in_row);
			PrintMatrix2D(Tlocal, myrows, mycols);
			printf("\n");
			//PrintMatrix2D(lastK, 2*bf, mycols);
			//printf("\n");
		    fflush(stdout);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	*/

	/*
	 * update last level of the table
	 */
		// last level (l=0)
		if (mpi_rank_row_in_col==0) 								// first row of procs
		{
			MPI_Bcast (&bb[0][0], m, MPI_DOUBLE, 0, comm_row);

			for (i=0; i<myxxrows; i++)
			{
				gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
				for(rhs=0;rhs<m;rhs++)
				{
					xx[gi][rhs]=xx[gi][rhs]+Tlocal[0][i]*bb[0][rhs];
				}
			}

			//MPI_Wait(&mpi_request, &mpi_status);

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

	/*
	for (i=0; i<cprocs; i++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		if (mpi_rank==i)
		{
			printf("[0]\n xx %d (%d,%d):\n",mpi_rank,mpi_rank_row_in_col,mpi_rank_col_in_row);
			PrintMatrix2D(xx, n, m);
			printf("\n");
		    fflush(stdout);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	*/

	//MPI_Wait(&mpi_request, &mpi_status);
	//MPI_Barrier(comm);

	// cleanup
	NULLFREE(lastKc);
	NULLFREE(lastKr);

	DeallocateMatrix2D(lastK,2*bf,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Tlocal,myrows,CONTIGUOUS);

	result.total_end_time = time(NULL);

	return result;
}
