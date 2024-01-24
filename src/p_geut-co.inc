
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
__attribute__((always_inline)) inline void pbDGEUT_CO (	int mpi_rank_row_in_col, int mpi_rank_col_in_row, int myrows, int mycols, int m,
														int l, int l_owner, int l_1_owner, int l_row, int l_col,
														int last_row,
														double* lastKr, double* lastKc, double* h, double** Tlocal)
*/
{
	int i,j;
    PRECISION hh;

	if (mpi_rank_col_in_row == l_owner)
	{
		if ( unlikely ( mpi_rank_col_in_row == mpi_rank_row_in_col ) )		// proc holding l-th col AND ON the diagonal
		{
			for (i=0; i < last_row; i++)
			{
				hh  = lastKc[i]*h[i];

				// before diagonal
				#pragma GCC ivdep
				for (j=0; j < i; j++)
				{
					Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[j]*hh;
				}

				// on diagonal
				Tlocal[i][j] = Tlocal[i][j]*h[i];

				// after diagonal but before l-th
				#pragma GCC ivdep
				for (j=i+1; j < l_col; j++)
				{
					Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[j]*hh;
				}

				// on l-th
				Tlocal[i][j] = - lastKr[j]*hh;

				// after l-th
				#pragma GCC ivdep
				for (j=l_col+1; j < mycols; j++)
				{
					Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[j]*hh;
				}
			}
		}
		else												// procs holding l-th col AND OFF the diagonal
		{
			for (i=0; i < last_row; i++)
			{
				hh  = lastKc[i]*h[i];

				// before l-th
				#pragma GCC ivdep
				for (j=0; j < l_col; j++)
				{
					Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[j]*hh;
				}

				// on l-th
				Tlocal[i][j] = - lastKr[j]*hh;

				// after l-th
				#pragma GCC ivdep
				for (j=l_col+1; j < mycols; j++)
				{
					Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[j]*hh;
				}
			}
		}
	}
	else if ( unlikely ( mpi_rank_col_in_row == mpi_rank_row_in_col ) )	// procs ON the diagonal but NOT holding the l-th col
	{
		for (i=0; i < last_row; i++)
		{
			hh  = lastKc[i]*h[i];

			// before diagonal
			#pragma GCC ivdep
			for (j=0; j < i; j++)
			{
				Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[j]*hh;
			}

			// on diagonal
			Tlocal[i][j] = Tlocal[i][j]*h[i];

			// after diagonal
			#pragma GCC ivdep
			for (j=i+1; j < mycols; j++)
			{
				Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[j]*hh;
			}
		}
	}
	else													// procs OFF the diagonal but NOT holding the l-th col
	{
		for (i=0; i < last_row; i++)
		{
			hh  = lastKc[i]*h[i];

			#pragma GCC ivdep
			for (j=0; j < mycols; j++)
			{
				Tlocal[i][j] = Tlocal[i][j]*h[i] - lastKr[j]*hh;
			}
		}
	}
}

#undef PRECISION
#undef MPI_PRECISION
