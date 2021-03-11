#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "helpers/macros.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "DGEZR.h"
#include "pbDGEIT_CX.bf1.h"


test_output pbDGESV_CO_dev(double** A, double** bb, double** xx, test_input input, parallel_env env, MPI_Comm comm)
{
	/*
	 * nb	NOT USED: blocking factor: number of adjacent column (block width)
	 * n	size (number of columns) of the square matrix A
	 * m	number of rigth-hand-sides (number of columns) in bb
	 *
	 */

	test_output result;

	result.total_start_time = time(NULL);

	MPI_Comm comm_calc;
	MPI_Comm comm_row;
	MPI_Comm comm_col;

	int i_spare;
    int cprocrows;
    int cproccols;
    int mpi_row;
    int mpi_col;

	if (env.mpi_rank < input.calc_procs)
	{
		i_spare = 0;
		MPI_Comm_split(comm, i_spare, env.mpi_rank, &comm_calc);

	    cprocrows = sqrt(input.calc_procs);
	    cproccols = cprocrows;

	    mpi_row = env.mpi_rank / cproccols;
	    mpi_col = env.mpi_rank % cproccols;
	}
	else
	{
		i_spare = 1;
		MPI_Comm_split(comm, i_spare, env.mpi_rank, &comm_calc);

	    cprocrows = sqrt(input.calc_procs);
	    cproccols = input.spare_procs / cprocrows;

	    mpi_row = (env.mpi_rank-input.calc_procs) / cproccols;
	    mpi_col = cprocrows + ((env.mpi_rank-input.calc_procs) % cproccols);
	}
	MPI_Comm_split(comm, mpi_row, env.mpi_rank, &comm_row);
	MPI_Comm_split(comm, mpi_col, env.mpi_rank, &comm_col);

	int mpi_rank_col_in_row;
	int mpi_rank_row_in_col;
	MPI_Comm_rank(comm_row, &mpi_rank_col_in_row);		// get current process id in row
	MPI_Comm_rank(comm_col, &mpi_rank_row_in_col);		// get current process id in col

	int i;

	for (i=0; i<input.calc_procs+input.spare_procs; i++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		if (env.mpi_rank==i)
		{
			printf("%d@(%d,%d) = (%d,%d)\n",env.mpi_rank,mpi_row,mpi_col,mpi_rank_row_in_col,mpi_rank_col_in_row);

			fflush(stdout);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}


	result.total_end_time = time(NULL);

	return result;
}
