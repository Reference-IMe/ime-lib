#include <mpi.h>
#include "helpers/macros.h"

/*
 * 	distributed
 *	init (IT) inhibition table T
 *	from general (GE) system matrix A of doubles (D)
 *
 */

#ifndef __pbDGEIT_CX_H__
#define __pbDGEIT_CX_H__

void pbDGEIT_CX(double** A, double** Tlocal, double** lastK, int n, int bf, int cprocs,
				MPI_Comm comm, int rank,
				MPI_Comm comm_row, int rank_col_in_row,
				MPI_Comm comm_col, int rank_row_in_col,
				MPI_Status* mpi_status,
				MPI_Request* mpi_request)
{
	int i,j;

    int cprocrows = sqrt(cprocs);
    int cproccols = cprocrows;
    int myrows   = n/cprocrows;		// num of rows per process
    int mycols   = n/cproccols;		// num of cols per process

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

	// block of A to be extracted and sent
	MPI_Datatype A_block;
	MPI_Type_vector (myrows, mycols, n, MPI_DOUBLE, & A_block );
	// never used, no need to commit
	//MPI_Type_commit (& A_block);

	// block of A to be extracted and sent, properly resized for scattering
	MPI_Datatype A_block_resized;
	MPI_Type_create_resized (A_block, 0, mycols*sizeof(double), & A_block_resized);
	MPI_Type_commit (& A_block_resized);

	// block of A extracted, to be stored transposed as contiguous columns in T (K part)
	MPI_Datatype K_column;
	MPI_Type_vector (myrows, 1, mycols, MPI_DOUBLE, & K_column );
	// never used, no need to commit
	//MPI_Type_commit (& K_column);

	// block of A extracted, to be stored as contiguous columns in T (K part), properly resized for scattering
	MPI_Datatype K_column_contiguous_resized;
	MPI_Type_create_resized (K_column, 0, 1*sizeof(double), & K_column_contiguous_resized);
	MPI_Type_commit (& K_column_contiguous_resized);

	/*
	 * scatter 1 block to every proc in grid
	 */
		int disps[cprocs];
		int counts[cprocs];
		for (i=0; i<cprocrows; i++)
		{
			for (j=0; j<cproccols; j++)
			{
				// 1 A_block_resized
				counts[i*cproccols+j]=1;
				// displacements with reference to the size of A_block_resized
				// straight scattering:
				//disps[i*cproccols+j]=i*cproccols*myrows+j;
				// transposed scattering:
				disps[i*cproccols+j]=j*cproccols*myrows+i;
				//if (rank==0) printf("(%d,%d)=%d %d\n",i,j,disps[i*cproccols+j],disps[i*cproccols+j]*mycols);
			}
		}
		MPI_Iscatterv (&A[0][0], counts, disps, A_block_resized, &Tlocal[0][0], mycols, K_column_contiguous_resized, 0, comm, &mpi_request[2]);

	/*
	 * distribute diagonal elements and init T table
	 */
		if (rank_row_in_col==rank_col_in_row)	// procs on main diagonal have to prepare buffer for broadcasting diagonal elements
		{
			// block of A must be here (contains diagonal elements)
			MPI_Wait( &mpi_request[0], &mpi_status[2]);

			// distribute
			for (i=0; i<mycols; i++)
			{
				lastKc[0][i]=Tlocal[i][i];
			}
			MPI_Ibcast (&lastKc[0][0], mycols, MPI_DOUBLE, rank_row_in_col, comm_row, &mpi_request[3]); // proc on main diagonal broadcasts to his row of procs

			// init
			for (i=0;i<myrows;i++)
			{
				for (j=0;j<i;j++)
				{
					Tlocal[i][j]=Tlocal[i][j]/lastKc[0][i];
				}

				Tlocal[i][i]=1/lastKc[0][i];

				for (j=i+1;j<mycols;j++)
				{
					Tlocal[i][j]=Tlocal[i][j]/lastKc[0][i];
				}
			}
		}
		else
		{
			// receive
			MPI_Ibcast (&lastKc[0][0], mycols, MPI_DOUBLE, rank_row_in_col, comm_row, &mpi_request[3]); // proc on main diagonal broadcasts to his row of procs

			// block of A and diagonal (lastKc) must be here before init
			MPI_Waitall(2, &mpi_request[2], &mpi_status[2]);

			// init
			for (i=0;i<myrows;i++)
			{
				for (j=0;j<mycols;j++)
				{
					Tlocal[i][j]=Tlocal[i][j]/lastKc[0][i];
				}
			}
		}

	//TODO: init last rows and cols at first, and then sync them immediately
	/*
	 *  sync last rows and cols
	 *
	 *  use the same request slots as in the main body, for proper waiting
	 */
		if (rank_row_in_col==cprocrows-1)
		{
			for (i=0; i<bf; i++)
			{
				for (j=0; j<mycols; j++)
				{
					lastKr[i][j]=Tlocal[myrows-bf][j];
				}
			}
			MPI_Ibcast ( &lastKr[0][0], bf*mycols, MPI_DOUBLE, cprocrows-1, comm_col, &mpi_request[1]);
		}
		else
		{
			MPI_Ibcast ( &lastKr[0][0], bf*mycols, MPI_DOUBLE, cprocrows-1, comm_col, &mpi_request[1]);
		}

		if (rank_col_in_row==cproccols-1)
		{
			for (i=0; i<myrows; i++)
			{
				for (j=0; j<bf; j++)
				{
					lastKc[j][i]=Tlocal[i][mycols-bf];
				}
			}
			MPI_Ibcast ( &lastKc[0][0], bf*myrows, MPI_DOUBLE, cproccols-1, comm_row, &mpi_request[3]);
		}
		else
		{
			MPI_Ibcast ( &lastKc[0][0], bf*myrows, MPI_DOUBLE, cproccols-1, comm_row, &mpi_request[3]);
		}
}

#endif
