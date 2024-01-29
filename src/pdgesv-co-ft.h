#ifndef __PDGESV_CO_FT_H__
#define __PDGESV_CO_FT_H__

#include <mpi.h>
#include <omp.h>
#include <time.h>

#include "../../../../testers/tester_structures.h"
#include "_gezr.h"
#include "helpers/macros.h"
#include "helpers/matrix_basic.h"
#include "helpers/vector_basic.h"
#include "pdgeit-c_-ft.h"
#include "pdgeub-co.h"
#include "pdgeuh-co.h"
#include "pdgeut-co.h"
#include "pdgeux-co.h"

/*
 *	parallel (P) solve (SV) system with general (GE) matrix A of doubles (D)
 *	of order n and with m r.h.s in matrix bb and solutions in xx
 *	with compact overwrite (CO) memory model
 *	with optimized initialization
 *	with fault tolerance (FT)
 *
 */

exit_status pdgesv_co_ft (	double** A,
							double** bb,
							double** xx,
							test_input input,
							parallel_env env,
							int num_of_failing_ranks,
							int* failing_rank_list,
							int failing_level,
							int recovery_enabled		)
{
	/*
	 * nb	NOT USED: blocking factor: number of adjacent column (block width)
	 * n	size (number of columns) of the square matrix A
	 * m	number of right-hand-sides number of columns) in bb
	 *
	 */
	#define TYPE REAL_DOUBLE
	#include "p_gesv-co-ft.inc"
	#undef TYPE
}

#endif
