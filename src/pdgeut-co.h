#ifndef __PDGEUT_CO_H__
#define __PDGEUT_CO_H__

#include <mpi.h>

__attribute__((always_inline)) inline void pdgeut_co (	int mpi_rank_row_in_col,
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
														double* lastKr,
														double* lastKc,
														double* h,
														double** Tlocal			)
{
	#define TYPE REAL_DOUBLE
	#include "p_geut-co.inc"
	#undef TYPE
}
#endif
