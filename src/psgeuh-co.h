#ifndef __PSGEUH_CO_H__
#define __PSGEUH_CO_H__

__attribute__((always_inline)) inline void psgeuh_co (	int mpi_rank_row_in_col,
														int mpi_rank_col_in_row,
														int myrows,
														int mycols,
														int m,
														int l,
														int l_owner,
														int l_1_owner,
														int l_row,
														int l_col,
														int last_row,
														float* lastKr,
														float* lastKc,
														float* h				)
{
	#include "p_geuh-co.inc"
}
#endif
