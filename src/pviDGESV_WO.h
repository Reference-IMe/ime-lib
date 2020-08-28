#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "helpers/macros.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "testers/tester_structures.h"


extern test_output pviDGESV_WO_u1ae(int nb, int n, double** A, int m, double** bb, double** xx, MPI_Comm comm);
/*
 *	solve (SV) system with general (GE) matrix A of doubles (D)
 *	of order n and with m r.h.s in matrix bb[n,m] and solutions in xx[n,m]
 *	with:
 *	wide overwrite (WO) memory model
 *	parallelized in interleaved columns (pvi) over cprocs calculating processors
 *	parallelized initialization
 *	optimized loops
 *	ifs removed
 *
 */
test_output pviDGESV_WO_default(int nb, int n, double** A, int m, double** bb, double** xx, MPI_Comm comm)
{
	return pviDGESV_WO_u1ae(nb, n, A, m, bb, xx, comm);
}
