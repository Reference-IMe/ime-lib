#ifndef __PSGEIT_C_FT_H__
#define __PSGEIT_C_FT_H__

#include <mpi.h>

#include "helpers/macros.h"

/*
 * 	distributed
 *	init (IT) inhibition table T
 *	from general (GE) system matrix A of floats (S)
 *	with fault tolerance
 *
 */

void psgeit_c_ft (	float** A,
					float** Tlocal,
					float* lastKr,
					float* lastKc,
					float** w,
					int n,
					int cprocs,
					int sproccols,
					MPI_Comm comm,
					int rank,
					MPI_Comm comm_row,
					int rank_col_in_row,
					MPI_Comm comm_col,
					int rank_row_in_col,
					MPI_Comm comm_row_calc,
					MPI_Status* mpi_status,
					MPI_Request* mpi_request	)
{
	#define TYPE REAL_SINGLE
	#include "p_geit-c_-ft.inc"
	#undef TYPE
	}

#endif
