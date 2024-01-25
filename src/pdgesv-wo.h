#ifndef __PDGESV_WO_H__
#define __PDGESV_WO_H__

#include <mpi.h>
#include <omp.h>
#include <time.h>

#include "helpers/macros.h"
#include "helpers/vector.h"
#include "../../../../testers/tester_structures.h"
#include "_gezr.h"
#include "helpers/matrix_basic.h"
#include "pdgeit-c_.h"

/*
 *	parallel (P) solve (SV) system with general (GE) matrix A of doubles (D)
 *	of order n and with m r.h.s in matrix bb and solutions in xx
 *	with wide overwrite (WO) memory model
 *	with optimized initialization
 *
 */

test_output pdgesv_wo (	int nb,
						int n,
						double** A,
						int m,
						double** bb,
						double** xx,
						MPI_Comm comm	)
{
	/*
	 * nb	NOT USED: blocking factor: number of adjacent column (block width)
	 * n	size (number of columns) of the square matrix A
	 * m	number of rigth-hand-sides (number of columns) in bb
	 *
	 */
	#define TYPE REAL_DOUBLE
	#include "p_gesv-wo.inc"
	#undef TYPE
}
