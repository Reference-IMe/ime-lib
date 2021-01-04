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
	MPI_Comm_split(comm, mpi_row, MPI_UNDEFINED, &comm_row);
	MPI_Comm_split(comm, mpi_col, MPI_UNDEFINED, &comm_col);

	int mpi_rank_row;
	int mpi_rank_col;
	MPI_Comm_rank(comm_row, &mpi_rank_row);		// get current process id in row
	MPI_Comm_rank(comm_col, &mpi_rank_col);		// get current process id in col

	MPI_Status  mpi_status;
	MPI_Request mpi_request = MPI_REQUEST_NULL;

	int i,j,l;						// general indexes
    int myrows   = n/cprocrows;		// num of rows per process
    int mycols   = n/cproccols;		// num of cols per process
    int myxxrows = mycols;			// num of chunks for better code readability
    int rhs;

    printf("rank %d: in (%d,%d) has ranks (%d,%d) - size %dx%d\n",mpi_rank,mpi_row,mpi_col,mpi_rank_row,mpi_rank_col,myrows,mycols);
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
	pbDGEIT_CX(A, Tlocal, lastK, n, nb, comm, mpi_rank, comm_row, mpi_rank_row, comm_col, mpi_rank_col, cprocs);	// init inhibition table

	for (i=0; i<cprocs; i++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		if (mpi_rank==i)
		{
			printf("Tlocal and lastK in %d (%d,%d):\n",mpi_rank,mpi_rank_row,mpi_rank_col);
			PrintMatrix2D(Tlocal, myrows, mycols);
			printf("\n");
			PrintMatrix2D(lastK, 2*nb, mycols);
			printf("\n");
		    fflush(stdout);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);



	/*
     * MPI derived types
     */
	//
	MPI_Datatype s_lastKr_chunk;
	MPI_Type_vector (nb, 1, mycols, MPI_DOUBLE, & s_lastKr_chunk );
	MPI_Type_commit (& s_lastKr_chunk);

	MPI_Datatype lastKr_chunk;
	MPI_Type_vector (nb, 1, n, MPI_DOUBLE, & lastKr_chunk );
	MPI_Type_commit (& lastKr_chunk);

	// proper resizing for gathering
	MPI_Datatype s_lastKr_chunk_resized;
	MPI_Type_create_resized (s_lastKr_chunk, 0, 1*sizeof(double), & s_lastKr_chunk_resized);
	MPI_Type_commit (& s_lastKr_chunk_resized);

	MPI_Datatype lastKr_chunk_resized;
	MPI_Type_create_resized (lastKr_chunk, 0, 1*sizeof(double), & lastKr_chunk_resized);
	MPI_Type_commit (& lastKr_chunk_resized);

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


	// last proc has already sent a full chunk of lastKr
	// decrease chunk size by nb for next sending
	gather_count[cprocs-1]=mycols-nb;



	MPI_Bcast (&bb[0][0], n*m, MPI_DOUBLE, 0, comm);								// send all r.h.s to all procs

	/*
	 *  calc inhibition sequence
	 */
	result.core_start_time = time(NULL);

	// general bounds for the loops over the columns
	// they differ on processes and change along the main loop over the levels
	int myxxstart = mycols;		// beginning column position for updating the solution (begins from right)
	int current_last=nb-1;		// index for the current last row or col of K in buffer
	int firstdiag=mpi_rank_row*mycols;	// (global) position of the first diagonal element on this mpi_rank
	int l_col;					// (local) position of the column l
	int l_owner;				// rank holding the column l
	int gi;						// global index
	//TODO: pre-calc other values..

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{

		for (i=0; i<cprocs; i++)
		{
			MPI_Barrier(MPI_COMM_WORLD);
			if (mpi_rank==i)
			{
				printf("Tlocal in %d (%d,%d):\n",mpi_rank,mpi_rank_row,mpi_rank_col);
				PrintMatrix2D(Tlocal, myrows, mycols);
				printf("\n");
			    fflush(stdout);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);


		//TODO: avoid if?
		if (mpi_rank_row==PVMAP(l, mycols)) // if a process contains the last rows/cols, must skip it
		{
			myxxstart--;
		}

		/*
		// update solutions
		// l .. n-1
		for (i=myxxstart; i<=PVLOCAL(n-1, mycols); i++)
		{
			gi=PVGLOBAL(i, mycols, mpi_rank_row);
			for (rhs=0;rhs<m;rhs++)
			{
				// on column < l X is null and not stored in T
				//if (global[i]>=l) xx[global[i]][rhs]=xx[global[i]][rhs]+Xlocal[l][i]*bb[l][rhs];
				xx[gi][rhs]=xx[gi][rhs]+Xlocal[l][i]*bb[l][rhs];
			}
		}
		*/

		// wait for new last rows and cols before computing helpers
		if (current_last==nb-1) MPI_Wait(&mpi_request, &mpi_status);
		// TODO: check performance penalty by skipping MPI_wait with an 'if' for non due cases (inside the blocking factor) or not

		// update helpers
		if (mpi_rank_col < PVMAP(l, myrows))
		{
			for (i=0; i<myrows; i++)
			{
				h[i]   = 1/(1-lastKc[current_last][i]*lastKr[current_last][i]);
				hh[i]  = lastKc[current_last][i]*h[i];
				for (rhs=0;rhs<m;rhs++)
				{
					bb[i][rhs] = bb[i][rhs]-lastKr[current_last][i]*bb[l][rhs];
				}
			}
		}
		else if (mpi_rank_col == PVMAP(l, myrows))
		{
			for (i=0; i<=( (l-1) % myrows); i++)
			{
				h[i]   = 1/(1-lastKc[current_last][i]*lastKr[current_last][i]);
				hh[i]  = lastKc[current_last][i]*h[i];
				for (rhs=0;rhs<m;rhs++)
				{
					bb[i][rhs] = bb[i][rhs]-lastKr[current_last][i]*bb[l][rhs];
				}
			}
		}

		for (i=0; i<cprocs; i++)
		{
			MPI_Barrier(MPI_COMM_WORLD);
			if (mpi_rank==i)
			{
				printf("h and hh in %d (%d,%d):\n",mpi_rank,mpi_rank_row,mpi_rank_col);
				PrintVector(h, myrows);
				printf("\n");
				PrintVector(hh, myrows);
				printf("\n");
			    fflush(stdout);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);




		l_col=PVLOCAL(l, mycols);
		int li;

		// must differentiate topological formula on special column l
		//
		if (mpi_rank_row==PVMAP(l, mycols))			// proc. in row containing column l
		{//rows:
			// before first diagonal element
			for (i=0; i<firstdiag; i++)
			{//columns:
				if (mpi_rank_col==PVMAP(i, myrows))
				{
					li=i % myrows;
					// before column l (K values)
					for (j=0; j<l_col; j++)
					{
						Klocal[li][j]=Klocal[li][j]*h[li] - Klocal[l][j]*hh[li];
					}

					// column l (X value)
					Xlocal[li][l_col]= - Xlocal[l][l_col]*hh[li];

					// after column l (X values)
					for (j=l_col+1; j<mycols; j++)
					{
						Xlocal[li][j]=Xlocal[li][j]*h[li] - Xlocal[l][j]*hh[li];
					}
				}
			}
			// containing first diagonal element
			for (i=firstdiag; i<firstdiag+l_col; i++)
			{//columns:
				if (mpi_rank_col==PVMAP(i, myrows))
				{
					li=i % myrows;
					// before diagonal element (K values)
					for (j=0; j<(i-firstdiag); j++)
					{
						Klocal[li][j]=Klocal[li][j]*h[li] - Klocal[l][j]*hh[li];
					}

					// diagonal element (X value)
					Xlocal[li][i-firstdiag]=Xlocal[li][i-firstdiag]*h[li];

					// // after diagonal element and before column l (K values)
					for (j=i-firstdiag+1; j<l_col; j++)
					{
						Klocal[li][j]=Klocal[li][j]*h[li] - Klocal[l][j]*hh[li];
					}

					// column l (X value)
					Xlocal[li][l_col]= - Xlocal[l][l_col]*hh[li];

					// after column l (X values)
					for (j=l_col+1; j<mycols; j++)
					{
						Xlocal[li][j]=Xlocal[li][j]*h[li] - Xlocal[l][j]*hh[li];
					}
				}
			}
			// remaining
			for (i=firstdiag+l_col; i<=l-1; i++)
			{
				if (mpi_rank_col==PVMAP(i, myrows))
				{
					li=i % myrows;
					// before column l (K values)
					for (j=0; j<l_col; j++)
					{
						Klocal[li][j]=Klocal[li][j]*h[li] - Klocal[l][j]*hh[li];
					}

					// column l (X value)
					Xlocal[li][l_col]= - Xlocal[l][l_col]*hh[li];

					// after column l (X values)
					for (j=l_col+1; j<mycols; j++)
					{
						Xlocal[li][j]=Xlocal[li][j]*h[li] - Xlocal[l][j]*hh[li];
					}
				}
			}
		}
		else										// proc. NOT containing column l
		{//rows:
			int very_last_row;
			very_last_row=MIN((firstdiag+mycols),l);

			// before first diagonal element
			for (i=0; i<MIN(firstdiag,l); i++)
			{//columns:
				if (mpi_rank_col==PVMAP(i, myrows))
				{
					li=i % myrows;
					// before column l (K values)
					for (j=0; j<l_col; j++)
					{
						Klocal[li][j]=Klocal[li][j]*h[li] - Klocal[l][j]*hh[li];
					}

					// after column l (X values)
					for (j=l_col; j<mycols; j++)
					{
						Xlocal[li][j]=Xlocal[li][j]*h[li] - Xlocal[l][j]*hh[li];
					}
				}
			}
			// containing first diagonal element
			for (i=firstdiag; i<very_last_row; i++)
			{//columns:
				if (mpi_rank_col==PVMAP(i, myrows))
				{
					li=i % myrows;
					// before diagonal element (K values)
					for (j=0; j<(i-firstdiag); j++)
					{
						Klocal[li][j]=Klocal[li][j]*h[li] - Klocal[l][j]*hh[li];
					}

					// diagonal element (X value)
					Xlocal[li][i-firstdiag]=Xlocal[li][i-firstdiag]*h[li];

					// after diagonal element (K values)
					for (j=i-firstdiag+1; j<mycols; j++)
					{
						Klocal[li][j]=Klocal[li][j]*h[li] - Klocal[l][j]*hh[li];
					}
				}
			}
			// remaining
			for (i=very_last_row; i<=l-1; i++)
			{
				if (mpi_rank_col==PVMAP(i, myrows))
				{
					li=i % myrows;
					// before column l (K values)
					for (j=0; j<l_col; j++)
					{
						Klocal[li][j]=Klocal[li][j]*h[li] - Klocal[l][j]*hh[li];
					}

					// after column l (X values)
					for (j=l_col; j<mycols; j++)
					{
						Xlocal[li][j]=Xlocal[li][j]*h[li] - Xlocal[l][j]*hh[li];
					}
				}
			}
		}

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

/*
	// last level (l=0)
	for (i=0; i<myxxrows; i++)
	{
		gi=PVGLOBAL(i, mycols, mpi_rank);
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
	if (mpi_rank==0)
	{
		MPI_Gather (MPI_IN_PLACE, m*myxxrows, MPI_DOUBLE, &xx[0][0], m*myxxrows, MPI_DOUBLE, 0, comm);
	}
	else
	{
		MPI_Gather (&xx[mpi_rank*myxxrows][0], m*myxxrows, MPI_DOUBLE, &xx[0][0], m*myxxrows, MPI_DOUBLE, 0, comm);
	}
*/
	MPI_Wait(&mpi_request, &mpi_status);
	MPI_Barrier(comm);

	// cleanup
	NULLFREE(lastKc);
	NULLFREE(lastKr);
	NULLFREE(gather_displacement);
	NULLFREE(gather_count);

	DeallocateMatrix2D(lastK,2*nb,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Tlocal,myrows,CONTIGUOUS);

	result.total_end_time = time(NULL);

	return result;
}
