#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "helpers/macros.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"

#include "pbDGESV_CO.dev.h"


/*
 *	solve (SV) system with general (GE) matrix A of doubles (D)
 *	of order n and with m r.h.s in matrix bb[n,m] and solutions in xx[n,m]
 *	with:
 *	compact overwrite (WO) memory model
 *	parallelized in NON-interleaved columns and rows (pb) over a grid of cprocs calculating processors
 *	parallelized initialization
 *	optimized loops
 *	ifs removed
 *
 */
test_output pbDGESV_CO_default(int nb, int n, double** A, int m, double** bb, double** xx, MPI_Comm comm)
{
	return EMPTY_OUTPUT;
}
