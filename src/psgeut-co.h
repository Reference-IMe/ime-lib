#ifndef __PSGEUT_CO_H__
#define __PSGEUT_CO_H__

#include <mpi.h>

__attribute__((always_inline)) inline void psgeut_co (	int mpi_rank_row_in_col,
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
														float* h,
														float** Tlocal			)
{
	#define TYPE REAL_SINGLE
	#include "p_geut-co.inc"
	#undef TYPE
}
#endif
