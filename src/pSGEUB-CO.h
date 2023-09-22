#include <mpi.h>
#include "helpers/macros.h"

#ifndef __pbSGEUB_CO_H__
#define __pbSGEUB_CO_H__

__attribute__((always_inline)) inline void pSGEUB_CO (	int mpi_rank_row_in_col, int mpi_rank_col_in_row, int myrows, int mycols, int m,
														int l, int l_owner, int l_1_owner, int l_row, int l_col,
														int l_1_col,
														float* lastKr, float** bb)
{
	#include "p_GEUB-CO.inc"
}
#endif
