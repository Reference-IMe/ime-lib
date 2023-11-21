#include <mpi.h>
#include <omp.h>
#include <time.h>

#include "_GEZR.h"
#include "helpers/macros.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "helpers/matrix_basic.h"
#include "pSGEIT-C_.h"


test_output pSGESV_WO_FT (	float** A, float** bb, float** xx,
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
	 * m	number of rigth-hand-sides (number of columns) in bb
	 *
	 */
	#define TYPE REAL_SINGLE
	#include "p_GESV-WO-FT.inc"
	#undef TYPE
}
