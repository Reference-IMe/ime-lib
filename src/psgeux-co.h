#ifndef __PSGEUX_CO_H__
#define __PSGEUX_CO_H__

#include <mpi.h>

__attribute__((always_inline)) inline void psgeux_co (	int mpi_rank_row_in_col,
														int mpi_rank_col_in_row,
														int myrows,
														int mycols,
														int m,
														int l,
														int l_owner,
														int l_1_owner,
														int l_row,
														int l_col,
														int* myxxstart,
														float* lastKr,
														float** xx,
														float** bb				)
{
	#include "p_geux-co.inc"
}
#endif
