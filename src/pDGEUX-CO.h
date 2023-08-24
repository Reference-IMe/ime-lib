#include <mpi.h>

#ifndef __pbDGEUX_CO_H__
#define __pbDGEUX_CO_H__

__attribute__((always_inline)) inline void pDGEUX_CO (	int mpi_rank_row_in_col, int mpi_rank_col_in_row, int myrows, int mycols, int m,
														int l, int l_owner, int l_1_owner, int l_row, int l_col,
														int* myxxstart,
														double* lastKr, double** xx, double** bb)
{
	#include "p_GEUX-CO.inc"
}
#endif
