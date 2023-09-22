#include <mpi.h>
#include "helpers/macros.h"

/*
 * 	distributed
 *	init (IT) inhibition table T
 *	from general (GE) system matrix A of floats (S)
 *
 */

#ifndef __pbSGEIT_CX_BF1_H__
#define __pbSGEIT_CX_BF1_H__

void pSGEIT_C (	float** A, float** Tlocal, float* lastKr, float* lastKc, int n, int cprocs,
				MPI_Comm comm, int rank,
				MPI_Comm comm_row, int rank_col_in_row,
				MPI_Comm comm_col, int rank_row_in_col,
				MPI_Status* mpi_status,
				MPI_Request* mpi_request)
{
	#define TYPE REAL_SINGLE
	#include "p_GEIT-C_.inc"
	#undef TYPE
}

#endif
