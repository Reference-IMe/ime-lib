/*
__attribute__((always_inline)) inline void pbDGEUB_CO (	int mpi_rank_row_in_col, int mpi_rank_col_in_row, int myrows, int mycols, int m,
														int l, int l_owner, int l_1_owner, int l_row, int l_col,
														int l_1_col,
														double* lastKr, double** bb)
*/
{
	int i,gi;						// general indexes
	int rhs;

	//if (mpi_rank_row_in_col==0) 							// first row of procs
	{
		// no need to "MPI_Wait": lastKr is already there

		if ( likely ( mpi_rank_col_in_row <= l_1_owner ) )
		{
			if ( likely ( mpi_rank_col_in_row < l_1_owner ) )			// procs before col l-1
			{
				for (i=0; i<mycols; i++)				// treat all cols
				{
					gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
					for (rhs=0;rhs<m;rhs++)
					{
						bb[gi][rhs] = bb[gi][rhs]-lastKr[i]*bb[l][rhs];
					}
				}
			}
			else // if ( mpi_rank_col_in_row == l_1_owner ) 		// proc holding col l-1
			{
				for (i=0; i<=l_1_col; i++)				// cols till l-1
				{
					gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
					for (rhs=0;rhs<m;rhs++)
					{
						bb[gi][rhs] = bb[gi][rhs]-lastKr[i]*bb[l][rhs];
					}
				}
			}
		}
		//else	{do nothing}							// procs after col l-1
	}
}
