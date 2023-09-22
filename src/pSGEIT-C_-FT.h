#include <mpi.h>
#include "helpers/macros.h"

/*
 * 	distributed
 *	init (IT) inhibition table T
 *	from general (GE) system matrix A of floats (D)
 *
 */

#ifndef __pbSGEIT_CX_BF1_FT_H__
#define __pbSGEIT_CX_BF1_FT_H__

void pSGEIT_C_FT (	float** A, float** Tlocal, float* lastKr, float* lastKc, float** w, int n, int cprocs,
					int sproccols,
					MPI_Comm comm, int rank,
					MPI_Comm comm_row, int rank_col_in_row,
					MPI_Comm comm_col, int rank_row_in_col,
					MPI_Comm comm_row_calc,
					MPI_Status* mpi_status,
					MPI_Request* mpi_request)
{
	#define TYPE REAL_SINGLE
	#include "p_GEIT-C_-FT.inc"
	#undef TYPE
	}

#endif
