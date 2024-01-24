/*
__attribute__((always_inline)) inline void pbDGEUH_CO (	int mpi_rank_row_in_col, int mpi_rank_col_in_row, int myrows, int mycols, int m,
														int l, int l_owner, int l_1_owner, int l_row, int l_col,
														int last_row,
														double* lastKr, double* lastKc, double* h)
*/
{
	int i;

	// chunks of last row and col are symmetrical to the main diagonal
	// they are hosted on procs on the diagonal
	// last_row = myrows | l_row | 0
	// last_row = ( mpi_rank_row_in_col < l_owner )*myrows + ( mpi_rank_row_in_col == l_owner )*l_row;

	#pragma GCC ivdep
	for (i=0; i<last_row; i++)
	{
		h[i]   = 1/(1-lastKc[i]*lastKr[i]);
	}
}
