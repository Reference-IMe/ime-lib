#include <mpi.h>
#include <omp.h>
#include <time.h>

#include "_GEZR.h"
#include "helpers/macros.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "helpers/matrix_basic.h"
#include "pDGEIT-C_-FT.h"
#include "pDGEUB-CO.h"
#include "pDGEUH-CO.h"
#include "pDGEUT-CO.h"
#include "pDGEUX-CO.h"

test_output pDGESV_CO_FT (	double** A, double** bb, double** xx,
							test_input input,
							parallel_env env,
							int num_of_failing_ranks,
							int* failing_rank_list,
							int failing_level,
							int recovery_enabled )
{
	/*
	 * nb	NOT USED: blocking factor: number of adjacent column (block width)
	 * n	size (number of columns) of the square matrix A
	 * m	number of right-hand-sides number of columns) in bb
	 *
	 */
	#define TYPE REAL_DOUBLE
	#include "p_GESV-CO-FT.inc"
	#undef TYPE
}
