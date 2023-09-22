#include "p_GEIT-C_-macros.h"

#if TYPE == REAL_DOUBLE
	#define PRECISION double
	#define MPI_PRECISION MPI_DOUBLE
#endif

#if TYPE == REAL_SINGLE
	#define PRECISION float
	#define MPI_PRECISION MPI_FLOAT
#endif

// #pragma message "building " __FILE__ " for type '" TOSTR( PRECISION ) "'"
/*
 * 	distributed
 *	init (IT) inhibition table T
 *	from general (GE) system matrix A of types PRECISION
 *
 */

/*
void pbDGEIT_CX_bf1_ft(double** A, double** Tlocal, double* lastKr, double* lastKc, double** w, int n, int cprocs,
				int sproccols,
				MPI_Comm comm, int rank,
				MPI_Comm comm_row, int rank_col_in_row,
				MPI_Comm comm_col, int rank_row_in_col,
				MPI_Comm comm_row_calc,
				MPI_Status* mpi_status,
				MPI_Request* mpi_request)
*/
{
	int i,j,l;

    int cprocrows = sqrt(cprocs);
    int cproccols = cprocrows;
    int myrows   = n/cprocrows;		// num of rows per process
    int mycols   = n/cproccols;		// num of cols per process

	PRECISION* diag;
			//diag=malloc(mycols*sizeof(PRECISION*));

	PRECISION** tmpTlocal;
			 tmpTlocal=FUNCNAME(AllocateMatrix2D_, PRECISION) (myrows, mycols, CONTIGUOUS);

	MPI_Comm comm_row_checksum;

	// block of A to be extracted and sent
	MPI_Datatype A_block;
	MPI_Type_vector (myrows, mycols, n, MPI_PRECISION, & A_block );
	// never used, no need to commit
	//MPI_Type_commit (& A_block);

	// block of A to be extracted and sent, properly resized for scattering
	MPI_Datatype A_block_resized;
	MPI_Type_create_resized (A_block, 0, mycols*sizeof(PRECISION), & A_block_resized);
	MPI_Type_commit (& A_block_resized);

	// block of A extracted, to be stored transposed as contiguous columns in T (K part)
	MPI_Datatype K_column;
	MPI_Type_vector (myrows, 1, mycols, MPI_PRECISION, & K_column );
	// never used, no need to commit
	//MPI_Type_commit (& K_column);

	// block of A extracted, to be stored as contiguous columns in T (K part), properly resized for scattering
	MPI_Datatype K_column_contiguous_resized;
	MPI_Type_create_resized (K_column, 0, 1*sizeof(PRECISION), & K_column_contiguous_resized);
	MPI_Type_commit (& K_column_contiguous_resized);

	if ( likely(rank < cprocs) )
	{
		diag=malloc(mycols*sizeof(PRECISION*));

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

		if ( likely(rank_row_in_col != cprocrows-1) )														// all rows of procs but the last one
		{
			if ( unlikely(rank_col_in_row == rank_row_in_col) )											// proc on the diagonal
			{
				// block of A must be here before distributing diagonal elements
				MPI_Wait( &mpi_request[2], &mpi_status[2]);

				// distribute diagonal elements to other procs in the row
				for (i=0; i<mycols; i++)
				{
					diag[i]=Tlocal[i][i];
					//TODO: reciprocal to avoid division in subsequent ops, any rounding concern?
				}
					// proc on main diagonal broadcasts to his row of procs
				MPI_Ibcast (&diag[0], mycols, MPI_PRECISION, rank_row_in_col, comm_row_calc, &mpi_request[0]);

				// diagonal elements are already here, then

				// init
				INIT_T_ON_DIAG(i, j, myrows, mycols, diag);

				// receive last rows
				MPI_Ibcast ( &lastKr[0], mycols, MPI_PRECISION, cprocrows-1, comm_col, &mpi_request[1]);

				// receive last cols
				MPI_Ibcast ( &lastKc[0], myrows, MPI_PRECISION, cproccols-1, comm_row, &mpi_request[3]);

			}
			else if ( unlikely(rank_col_in_row == cproccols-1) )											// last proc in the row
			{
				// receive diagonal elements
				MPI_Ibcast (&diag[0], mycols, MPI_PRECISION, rank_row_in_col, comm_row_calc, &mpi_request[0]);

				// block of A and diagonal (lastKc) must be here before init
				//MPI_Waitall(2, &mpi_request[2], &mpi_status[2]);
				MPI_Wait( &mpi_request[2], &mpi_status[2]);
				MPI_Wait( &mpi_request[0], &mpi_status[0]);

				// init
				INIT_T_OFF_DIAG(i, j, myrows, mycols, diag);

				// distribute last cols
				for (i=0; i<myrows; i++)
				{
					lastKc[i]=Tlocal[i][mycols-1];
				}
				MPI_Ibcast ( &lastKc[0], myrows, MPI_PRECISION, cproccols-1, comm_row, &mpi_request[3]);

				// receive last rows
				MPI_Ibcast ( &lastKr[0], mycols, MPI_PRECISION, cprocrows-1, comm_col, &mpi_request[1]);

			}
			else 																				// all procs in the row but the diagonal and the last one
			{
				// receive diagonal elements
				MPI_Ibcast (&diag[0], mycols, MPI_PRECISION, rank_row_in_col, comm_row_calc, &mpi_request[0]);

				// block of A and diagonal (lastKc) must be here before init
				//MPI_Waitall(2, &mpi_request[2], &mpi_status[2]);
				MPI_Wait( &mpi_request[2], &mpi_status[2]);
				MPI_Wait( &mpi_request[0], &mpi_status[0]);

				// init
				INIT_T_OFF_DIAG(i, j, myrows, mycols, diag);

				// receive last rows
				MPI_Ibcast ( &lastKr[0], mycols, MPI_PRECISION, cprocrows-1, comm_col, &mpi_request[1]);

				// receive last cols
				MPI_Ibcast ( &lastKc[0], myrows, MPI_PRECISION, cproccols-1, comm_row, &mpi_request[3]);

			}
		}
		else																						// the last row of procs
		{
			if ( unlikely(rank_col_in_row == cproccols-1) )												// last proc in the row (which is also on the diagonal)
			{
				// block of A must be here before distributing diagonal elements
				MPI_Wait( &mpi_request[2], &mpi_status[2]);

				// distribute diagonal elements to other procs in the row
				for (i=0; i<mycols; i++)
				{
					diag[i]=Tlocal[i][i];
				}
					// proc on main diagonal broadcasts to his row of procs
				MPI_Ibcast (&diag[0], mycols, MPI_PRECISION, cproccols-1, comm_row_calc, &mpi_request[0]);

				// diagonal elements are already here, then

				// init
				INIT_T_ON_DIAG(i, j, myrows, mycols, diag);

				// distribute last rows
				for (j=0; j<mycols; j++)
				{
					lastKr[j]=Tlocal[myrows-1][j];
				}
				MPI_Ibcast ( &lastKr[0], mycols, MPI_PRECISION, cprocrows-1, comm_col, &mpi_request[1]);

				// distribute last cols
				for (i=0; i<myrows; i++)
				{
						lastKc[i]=Tlocal[i][mycols-1];
				}
				MPI_Ibcast ( &lastKc[0], myrows, MPI_PRECISION, cproccols-1, comm_row, &mpi_request[3]);

			}
			else 																				// all procs in the row but the last one (which also on the diagonal)
			{
				// receive diagonal elements
				MPI_Ibcast (&diag[0], mycols, MPI_PRECISION, cproccols-1, comm_row_calc, &mpi_request[0]);

				// block of A and diagonal (lastKc) must be here before init
				//MPI_Waitall(2, &mpi_request[2], &mpi_status[2]);
				MPI_Wait( &mpi_request[2], &mpi_status[2]);
				MPI_Wait( &mpi_request[0], &mpi_status[0]);

				// init
				INIT_T_OFF_DIAG(i, j, myrows, mycols, diag);

				// distribute last rows
				for (j=0; j<mycols; j++)
				{
					lastKr[j]=Tlocal[myrows-1][j];
				}
				MPI_Ibcast ( &lastKr[0], mycols, MPI_PRECISION, cprocrows-1, comm_col, &mpi_request[1]);

				// receive last cols
				MPI_Ibcast ( &lastKc[0], myrows, MPI_PRECISION, cproccols-1, comm_row, &mpi_request[3]);
			}
		}

		MPI_Wait( &mpi_request[0], &mpi_status[0]);
		NULLFREE(diag);
	}
	else
	{
		// receive last cols
		MPI_Ibcast ( &lastKc[0], myrows, MPI_PRECISION, cproccols-1, comm_row, &mpi_request[3]);
	}


    /*
     * check weights
     */
    /*
    if (rank==0)
    {
    	printf("weights:\n");
    	PrintMatrix2D_double(w, cprocrows, sproccols );
    }
    */
    for (l=0; l<sproccols; l++)
    {
		if ( likely(rank_col_in_row < cproccols) )
		{
			for (i=0; i<myrows; i++)
			{
				#pragma GCC ivdep
				for (j=0; j<mycols; j++)
				{
					//tmpTlocal[i][j]=(rank_row_in_col+1)*100+rank_col_in_row+1;
					tmpTlocal[i][j]=Tlocal[i][j] * w[rank_col_in_row][l];
				}
			}
			if ( unlikely(rank_col_in_row == rank_row_in_col) )
			{
				#pragma GCC ivdep
				for (j=0; j<mycols; j++)
				{
					tmpTlocal[j][j]=tmpTlocal[j][j] + w[rank_col_in_row][l];
				}
			}
			MPI_Comm_split(comm_row, 1, rank, &comm_row_checksum);
			// destination has rank cproccols because the communicator is split every time and has always cproccols+1 processes
			MPI_Reduce( &tmpTlocal[0][0], NULL, myrows*mycols, MPI_PRECISION, MPI_SUM, cproccols, comm_row_checksum );
		}
		else
		{
			if ( unlikely(rank_col_in_row == (cproccols+l) ) )
			{
				for (i=0; i<myrows; i++)
				{
					#pragma GCC ivdep
					for (j=0; j<mycols; j++)
					{
						tmpTlocal[i][j]=0;
					}
				}
				MPI_Comm_split(comm_row, 1, rank, &comm_row_checksum);
				MPI_Reduce( &tmpTlocal[0][0], &Tlocal[0][0], myrows*mycols, MPI_PRECISION, MPI_SUM, cproccols, comm_row_checksum );

				if ( rank_row_in_col == cprocrows-1 )
				{
					#pragma GCC ivdep
					for (j=0; j<mycols; j++)
					{
						lastKr[j]=Tlocal[myrows-1][j];
					}
				}
				MPI_Ibcast ( &lastKr[0], mycols, MPI_PRECISION, cprocrows-1, comm_col, &mpi_request[1]);
			}
			else
			{
				MPI_Comm_split(comm_row, 0, rank, &comm_row_checksum);
				// does not participate in reduction
			}
		}
        MPI_Comm_free(&comm_row_checksum);
    }

    // cleanup
    FUNCNAME(DeallocateMatrix2D_, PRECISION)(tmpTlocal,myrows,CONTIGUOUS);
}

#undef PRECISION
#undef MPI_PRECISION
