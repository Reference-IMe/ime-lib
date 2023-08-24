/*
__attribute__((always_inline)) inline void pbDGEUX_CO (	int mpi_rank_row_in_col, int mpi_rank_col_in_row, int myrows, int mycols, int m,
														int l, int l_owner, int l_1_owner, int l_row, int l_col,
														int* myxxstart,
														double* lastKr, double** xx, double** bb)
*/
{
	int i,gi,rhs;						// general indexes

//	if (mpi_rank_row_in_col==0) 					// first row of procs
	{
		// if a process contains the l-th cols, must skip it
		/*
		if (mpi_rank_col_in_row==l_owner)
		{
			myxxstart--;
		}
		*/
		// avoid if
		*myxxstart = *myxxstart - (mpi_rank_col_in_row == l_owner);

		// bb[l] must be here
		//MPI_Wait( mpi_req_bb, mpi_st_bb);

		// lastKr must be here
		//MPI_Wait( mpi_req_row, mpi_st_row);

		// bb[l] (or full bb, for the first iteration after init) and lastKr must be here
		//MPI_Waitall(2, mpi_request, mpi_status);


		// update xx vector
		// l .. n-1
		for (i=*myxxstart; i<myrows; i++)
		{
			gi=PVGLOBAL(i, mycols, mpi_rank_col_in_row);
			#pragma GCC ivdep
			for (rhs=0;rhs<m;rhs++)
			{
				xx[gi][rhs]=xx[gi][rhs]+lastKr[i]*bb[l][rhs];
			}
		}
	}
}
