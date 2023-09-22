#include <mpi.h>
#include "helpers/macros.h"
#include "p_GEIT-C_.macros"

/*
 * 	distributed
 *	init (IT) inhibition table T
 *	from general (GE) system matrix A of doubles (D)
 *
 */

#ifndef __pbDGEIT_CX_BF1_FT_H__
#define __pbDGEIT_CX_BF1_FT_H__

void pDGEIT_C_FT (	double** A, double** Tlocal, double* lastKr, double* lastKc, double** w, int n, int cprocs,
					int sproccols,
					MPI_Comm comm, int rank,
					MPI_Comm comm_row, int rank_col_in_row,
					MPI_Comm comm_col, int rank_row_in_col,
					MPI_Comm comm_row_calc,
					MPI_Status* mpi_status,
					MPI_Request* mpi_request)
{
	#define TYPE REAL_DOUBLE
	#include "p_GEIT-C_-FT.inc"
	#undef TYPE
	}

#endif
