#include <mpi.h>
#include "helpers/macros.h"

/*
 * 	distributed
 *	init (IT) inhibition table T
 *	from general (GE) system matrix A of doubles (D)
 *
 */

#ifndef __pbDGEIT_CX_BF1_H__
#define __pbDGEIT_CX_BF1_H__

void pDGEIT_C (	double** A, double** Tlocal, double* lastKr, double* lastKc, int n, int cprocs,
				MPI_Comm comm, int rank,
				MPI_Comm comm_row, int rank_col_in_row,
				MPI_Comm comm_col, int rank_row_in_col,
				MPI_Status* mpi_status,
				MPI_Request* mpi_request)
{
	#define TYPE REAL_DOUBLE
	#include "p_GEIT-C_.inc"
	#undef TYPE
}

#endif
