#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "helpers/macros.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "DGEZR.h"
//#include "pvDGEIT_CX.h"
#include "pbDGEIT_CX.h"


test_output pbDGESV_CO_g_smallest(int nb, int n, double** A, int m, double** bb, double** xx, MPI_Comm comm)
{
	/*
	 * nb	blocking factor: number of adjacent column (block width)
	 * n	size (number of columns) of the square matrix A
	 * m	number of rigth-hand-sides (number of columns) in bb
	 *
	 */

	test_output result;

	result.total_start_time = time(NULL);

    int mpi_rank, cprocs; //
    MPI_Comm_rank(comm, &mpi_rank);		// get current process id
    MPI_Comm_size(comm, &cprocs);	// get number of processes

    int cprocrows = sqrt(cprocs);
    int cproccols = cprocrows;

    int mpi_row = mpi_rank / cproccols;
    int mpi_col = mpi_rank % cproccols;

    //printf("%d OK\n",mpi_rank);


	MPI_Comm comm_row;
	MPI_Comm comm_col;
	MPI_Comm_split(comm, mpi_row, mpi_rank, &comm_row);
	MPI_Comm_split(comm, mpi_col, mpi_rank, &comm_col);

	int mpi_rank_col_in_row;
	int mpi_rank_row_in_col;
	MPI_Comm_rank(comm_row, &mpi_rank_col_in_row);		// get current process id in row
	MPI_Comm_rank(comm_col, &mpi_rank_row_in_col);		// get current process id in col

	MPI_Status  mpi_status;
	MPI_Request mpi_request = MPI_REQUEST_NULL;

	int i,j,l;						// general indexes
    int myrows   = n/cprocrows;		// num of rows per process
    int mycols   = n/cproccols;		// num of cols per process
    int myxxrows = mycols;			// num of chunks for better code readability
    int rhs;

    printf("rank %d: in (%d,%d) has ranks (%d,%d) - size %dx%d\n",mpi_rank,mpi_row,mpi_col,mpi_rank_row_in_col,mpi_rank_col_in_row,myrows,mycols);
    fflush(stdout);

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Tlocal;
			 Tlocal=AllocateMatrix2D(myrows, mycols, CONTIGUOUS);
	// aliases for better code readability
    // X part
    double** Xlocal;
    		 Xlocal=Tlocal;
    // K part
	double** Klocal;
			 Klocal=Tlocal;

	// last rows and cols of K
	double** lastK;
			 lastK=AllocateMatrix2D(2*nb, mycols, CONTIGUOUS);	// last rows [0 - (bf-1)] and cols [ bf - (2bf -1)] of K
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
    		h=AllocateVector(myrows);
    double* hh;
			hh=AllocateVector(myrows);

	/*
	 *  init inhibition table
	 */
	DGEZR(xx, n, m);												// init (zero) solution vectors
	pbDGEIT_CX(A, Tlocal, lastK, n, nb, comm, mpi_rank, comm_row, mpi_rank_col_in_row, comm_col, mpi_rank_row_in_col, cprocs);	// init inhibition table

	/*
	for (i=0; i<cprocs; i++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		if (mpi_rank==i)
		{
			printf("Tlocal and lastK in %d (%d,%d):\n",mpi_rank,mpi_rank_row_in_col,mpi_rank_col_in_row);
			PrintMatrix2D(Tlocal, myrows, mycols);
			printf("\n");
			PrintMatrix2D(lastK, 2*nb, mycols);
			printf("\n");
		    fflush(stdout);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	*/

	if (mpi_rank_row_in_col==0) 								// first row of procs
	{
		MPI_Bcast (&bb[0][0], n*m, MPI_DOUBLE, 0, comm_row);		// send all r.h.s
	}


	/*
	 *  calc inhibition sequence
	 */
	result.core_start_time = time(NULL);

	// general bounds for the loops over the columns
	// they differ on processes and change along the main loop over the levels
	int myxxstart = mycols;		// beginning column position for updating the solution (begins from right)
	int current_last=nb-1;		// index for the current last row or col of K in buffer
	int l_col;					// (local) position of the column l
	int last_row;				// (local) last row to process
	int last_col;
	//TODO: pre-calc other values..

	int li;
	int gi,gj;

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{

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
				PrintMatrix2D(lastK, 2*nb, mycols);
				printf("\n");
			    fflush(stdout);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);


		if (mpi_rank_row_in_col==0) 								// first row of procs
		{
			if (mpi_rank_col_in_row==PVMAP(l, mycols)) // if a process contains the last rows/cols, must skip it
			{
				myxxstart--;
			}
			//TODO: avoid if?

			printf("\n[%d] %d: %d %d\n",l,mpi_rank_col_in_row,myxxstart,PVLOCAL(n-1, mycols));

			MPI_Bcast (&bb[l][0], m, MPI_DOUBLE, PVMAP(l, mycols), comm_row);

			// update solutions
			// l .. n-1
			for (i=myxxstart; i<=PVLOCAL(n-1, mycols); i++)
			{
				gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
				for (rhs=0;rhs<m;rhs++)
				{
					// on column < l X is null and not stored in T
					//if (global[i]>=l) xx[global[i]][rhs]=xx[global[i]][rhs]+Xlocal[l][i]*bb[l][rhs];
					xx[gi][rhs]=xx[gi][rhs]+lastKr[current_last][i]*bb[l][rhs];
				}
			}
		}


		// wait for new last rows and cols before computing helpers
		//if (current_last==nb-1) MPI_Wait(&mpi_request, &mpi_status);
		// TODO: check performance penalty by skipping MPI_wait with an 'if' for non due cases (inside the blocking factor) or not

		if (mpi_rank_row_in_col < PVMAP(l, myrows)) 		// all rows
		{
			last_row=myrows;
		}
		else if (mpi_rank_row_in_col == PVMAP(l, myrows))	// rows till l-1
		{
			last_row=PVLOCAL(l, myrows);
		}
		else
		{
			last_row=0;										// no rows
		}

		// update helpers
		if (mpi_rank_row_in_col==mpi_rank_col_in_row)
		{
			for (i=0; i<last_row; i++)
			{
				h[i]   = 1/(1-lastKc[current_last][i]*lastKr[current_last][i]);
				hh[i]  = lastKc[current_last][i]*h[i];
				/*
				for (rhs=0;rhs<m;rhs++)
				{
					bb[i][rhs] = bb[i][rhs]-lastKr[current_last][i]*bb[l][rhs];
				}
				*/
			}
		}
		MPI_Bcast ( &h[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row);
		MPI_Bcast ( &hh[0], last_row, MPI_DOUBLE, mpi_rank_row_in_col, comm_row);

		if (mpi_rank_col_in_row < PVMAP(l-1, mycols)) 		// all cols
		{
			last_col=mycols-1;
		}
		else if (mpi_rank_col_in_row == PVMAP(l-1, mycols))	// cols till l-1
		{
			last_col=PVLOCAL(l-1, mycols);
		}
		else
		{
			last_col=-1;										// no rows
		}

		if (mpi_rank_row_in_col==0) 								// first row of procs
		{
			//MPI_Bcast (&bb[l][0], m, MPI_DOUBLE, PVMAP(l, mycols), comm_row);

			for (i=0; i<=last_col; i++)
			{
				for (rhs=0;rhs<m;rhs++)
				{
					bb[mpi_rank_col_in_row*mycols+i][rhs] = bb[mpi_rank_col_in_row*mycols+i][rhs]-lastKr[current_last][i]*bb[l][rhs];
				}
			}
		}
		/*
		for (i=0; i<cprocs; i++)
		{
			MPI_Barrier(MPI_COMM_WORLD);
			if (mpi_rank==i)
			{
				printf("h and hh in %d (%d,%d):\n",mpi_rank,mpi_rank_row_in_col,mpi_rank_col_in_row);
				PrintVector(h, myrows);
				//printf("\n");
				//PrintVector(hh, myrows);
				printf("\n");
			    fflush(stdout);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		*/


		/*
		if (mpi_rank_row_in_col < PVMAP(l, myrows)) // all rows
		{
			last_row=myrows-1;
		}
		else if (mpi_rank_row_in_col == PVMAP(l, myrows))
		{
			last_row=(l-1) % myrows;
		}
		else
		{
			last_row=-1;
		}
		*/

		//for (i=0; i <= last_row; i++)
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

		l_col=PVLOCAL(l-1, mycols);

		// sync last col
		if (mpi_rank_row_in_col <= PVMAP(l-1, myrows))
		{
			if (mpi_rank_col_in_row == PVMAP(l-1, mycols))
			{
				//for (i=0; i <= last_row; i++)
				for (i=0; i < last_row; i++)
				{
					lastKc[current_last][i]=Tlocal[i][l_col];
				}
			}
			//printf("%d bcast from %d to %d (%d,%d)\n",last_row+1,PVMAP(l-1, mycols),mpi_rank,mpi_rank_row_in_col,mpi_rank_col_in_row);
			//MPI_Bcast ( &lastKc[current_last][0], last_row+1, MPI_DOUBLE, PVMAP(l-1, mycols), comm_row);
			MPI_Bcast ( &lastKc[current_last][0], last_row, MPI_DOUBLE, PVMAP(l-1, mycols), comm_row);
		}

		// sync last row
		if (mpi_rank_row_in_col == PVMAP(l-1, myrows))
		{
			for (i=0; i < mycols; i++)
			{
				lastKr[current_last][i]=Tlocal[l_col][i];
			}
		}
		MPI_Bcast ( &lastKr[current_last][0], mycols, MPI_DOUBLE, PVMAP(l-1, myrows), comm_col);


/*
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

			l_owner = PVMAP(l-nb, mycols);

			// collect chunks of last row of K to "future" last node
			// "current" last node sends smaller chunks until 0
			MPI_Igatherv (&Klocal[l-nb][0], gather_count[mpi_rank], s_lastKr_chunk_resized, &lastKr[0][0], gather_count, gather_displacement, lastKr_chunk_resized, l_owner, comm, &mpi_request);


			//future last node broadcasts last rows and cols of K
			if (mpi_rank==l_owner)
			{
				// copy data into local buffer before broadcast
				for(j=0;j<nb;j++)
				{
					for (i=0; i<=l-1; i++)
					{
						lastKc[j][i]=Klocal[i][PVLOCAL(l-nb, mycols)+j];
					}
				}
				// wait for gathering to complete
				MPI_Wait(&mpi_request, &mpi_status);
			}
			// do not wait all for gather: only who has to broadcast
			//MPI_Wait(&mpi_request, &mpi_status);
			MPI_Ibcast (&lastK[0][0], 2*n*nb, MPI_DOUBLE, l_owner, comm, &mpi_request);

			// decrease the size of next chunk from "current" last node
			gather_count[l_owner]=gather_count[l_owner]-nb;
		}
		*/

	}


	for (i=0; i<cprocs; i++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		if (mpi_rank==i)
		{
			printf("[0]\nTlocal and lastK in %d (%d,%d):\n",mpi_rank,mpi_rank_row_in_col,mpi_rank_col_in_row);
			PrintMatrix2D(Tlocal, myrows, mycols);
			printf("\n");
			//PrintMatrix2D(lastK, 2*nb, mycols);
			//printf("\n");
		    fflush(stdout);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);


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

	for (i=0; i<cprocs; i++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		if (mpi_rank==i)
		{
			printf("[0]\n xx %d (%d,%d):\n",mpi_rank,mpi_rank_row_in_col,mpi_rank_col_in_row);
			PrintMatrix2D(xx, n, m);
			printf("\n");
			//PrintMatrix2D(lastK, 2*nb, mycols);
			//printf("\n");
		    fflush(stdout);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	//MPI_Wait(&mpi_request, &mpi_status);
	//MPI_Barrier(comm);

	// cleanup
	/*
	NULLFREE(lastKc);
	NULLFREE(lastKr);
	NULLFREE(gather_displacement);
	NULLFREE(gather_count);

	DeallocateMatrix2D(lastK,2*nb,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Tlocal,myrows,CONTIGUOUS);
	*/
	result.total_end_time = time(NULL);

	return result;
}
