#include <mpi.h>

#ifndef __pbDGEUT_CO_H__
#define __pbDGEUT_CO_H__

__attribute__((always_inline)) inline void pDGEUT_CO (	int mpi_rank_row_in_col, int mpi_rank_col_in_row, int myrows, int mycols, int m,
														int l, int l_owner, int l_1_owner, int l_row, int l_col,
														int last_row,
														double* lastKr, double* lastKc, double* h, double** Tlocal)
{
	#define TYPE REAL_DOUBLE
	#include "p_GEUT-CO.inc"
	#undef TYPE
}
#endif
