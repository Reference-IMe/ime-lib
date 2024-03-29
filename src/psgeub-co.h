#ifndef __PSGEUB_CO_H__
#define __PSGEUB_CO_H__

#include <mpi.h>

#include "helpers/macros.h"

__attribute__((always_inline)) inline void psgeub_co (	int mpi_rank_row_in_col,
														int mpi_rank_col_in_row,
														int myrows,
														int mycols,
														int m,
														int l,
														int l_owner,
														int l_1_owner,
														int l_row,
														int l_col,
														int l_1_col,
														float* lastKr,
														float** bb					)
{
	#include "p_geub-co.inc"
}
#endif
