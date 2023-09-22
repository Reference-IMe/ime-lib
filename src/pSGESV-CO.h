#include <mpi.h>
#include <omp.h>
#include <time.h>

#include "constants.h"
#include "_GEZR.h"
#include "helpers/macros.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "helpers/matrix_basic.h"
#include "pSGEIT-C_.h"


test_output pSGESV_CO ( int nb, int n, float** A, int m, float** bb, float** xx, MPI_Comm comm)
{
	/*
	 * nb	NOT USED: blocking factor: number of adjacent column (block width)
	 * n	size (number of columns) of the square matrix A
	 * m	number of rigth-hand-sides (number of columns) in bb
	 *
	 */
	#define TYPE REAL_SINGLE
	#include "p_GESV-CO.inc"
	#undef TYPE
}
