#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "helpers/macros.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"
#include "DGEZR.h"
#include "helpers/matrix_basic.h"
#include "pDGEIT-C_.h"


test_output pDGESV_CO ( int nb, int n, double** A, int m, double** bb, double** xx, MPI_Comm comm)
{
	/*
	 * nb	NOT USED: blocking factor: number of adjacent column (block width)
	 * n	size (number of columns) of the square matrix A
	 * m	number of rigth-hand-sides (number of columns) in bb
	 *
	 */
	#define PRECISION double
	#define MPI_PRECISION MPI_DOUBLE
	#include "p_GESV-CO.inc"
}
