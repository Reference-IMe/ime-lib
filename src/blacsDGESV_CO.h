/*
 * blacsDGESV_CO.h
 *
 *  Created on: Jan 4, 2021
 *      Author: marcello
 *
 *      https://www.mathkeisan.com/UsersGuide/ScaLAPACK.cfm
 *
 *      Is there a way to extract the diagonal from a matrix with simple matrix operations
 *      https://math.stackexchange.com/questions/1527670/is-there-a-way-to-extract-the-diagonal-from-a-matrix-with-simple-matrix-operatio
 *      https://stackoverflow.com/questions/59007147/blas-routine-to-compute-diagonal-elements-only-of-a-matrix-product
 *
 *      rank-1 update
 *      http://www.netlib.org/scalapack/explore-html/d9/d15/pdger___8c_source.html
 *
 */

#include <mpi.h>
#include <time.h>
#include "helpers/macros.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "helpers/Cblacs.h"
#include "helpers/scalapack.h"
#include "testers/tester_structures.h"


test_output blacsDGESV_CO(int n, double* A_global, int m, double* B_global, int nb,
								int mpi_rank, int cprocs,
								int nprow, int npcol, int myrow, int mycol,
								int context, int context_global)
{
	test_output result = EMPTY_OUTPUT;

	result.total_start_time = time(NULL);

	/*
	 * n = system rank (A_global n x n)
	 * m = num. of r.h.s (B_global n x m)
	 */

	// general
	int i,j;
	int i0 = 0;
	int i1 = 1;
	double d0 = 0.0;
	double d1 = 1.0;
	int info;
	int *ipiv;

	// matrix
	int nr, nc;
	double *A;
	double *At;
	double *Tmp1;
	double *Tmp2;
	double *J;

	int ncrhs, nrrhs;
	double *B;
	int ncrhst, nrrhst;
	double *Bt;

	int descA_global[9];
	int descB_global[9];
	int descA[9];
	int descAt[9];
	int descTmp1[9];
	int descTmp2[9];
	int descJ[9];

	int descB[9];
	int descBt[9];

	int lld, lldt;

	int ncj, nrj;

	if (mpi_rank < cprocs)
	{
		// Computation of local matrix size
		nc = numroc_( &n, &nb, &mycol, &i0, &npcol );
		nr = numroc_( &n, &nb, &myrow, &i0, &nprow );
		lld = MAX( 1 , nr );
		A  = malloc(nr*nc*sizeof(double));
		At = malloc(nr*nc*sizeof(double));
		Tmp1 = malloc(nr*nc*sizeof(double));
		Tmp2 = malloc(nr*nc*sizeof(double));

		ncj = numroc_( &n, &nb, &mycol, &i0, &npcol );
		nrj = numroc_( &i1,  &nb, &myrow, &i0, &nprow );
		J = malloc(ncj*nrj*sizeof(double));

		ncrhs = numroc_( &m, &nb, &mycol, &i0, &npcol );
		nrrhs = numroc_( &n, &nb, &myrow, &i0, &nprow );
		B = malloc(nrrhs*ncrhs*sizeof(double));

		ncrhst = numroc_( &n, &nb, &mycol, &i0, &npcol );
		nrrhst = numroc_( &m, &nb, &myrow, &i0, &nprow );
		lldt = MAX( 1 , nrrhst );
		Bt = malloc(nrrhst*ncrhst*sizeof(double));

		ipiv = malloc((lld+nb)*sizeof(int));

		// Descriptors (local)
		descinit_( descA, &n, &n, &nb, &nb, &i0, &i0, &context, &lld, &info );
		descinit_( descAt, &n, &n, &nb, &nb, &i0, &i0, &context, &lld, &info );

		descinit_( descTmp1, &n, &n, &nb, &nb, &i0, &i0, &context, &lld, &info );
		descinit_( descTmp2, &n, &n, &nb, &nb, &i0, &i0, &context, &lld, &info );
		descinit_( descJ, &i1, &n, &nb, &nb, &i0, &i0, &context, &i1, &info );

		descinit_( descB, &n, &m, &nb, &nb, &i0, &i0, &context, &lld, &info );
		descinit_( descBt, &m, &n, &nb, &nb, &i0, &i0, &context, &lldt, &info );

		if (mpi_rank==0)
		{
			// Descriptors (global)
			descinit_( descA_global, &n, &n, &i1, &i1, &i0, &i0, &context_global, &n, &info );
			descinit_( descB_global, &n, &m, &i1, &i1, &i0, &i0, &context_global, &n, &info );
		}
		else
		{
			// Descriptors (global, for non-root nodes)
			for (i=0; i<9; i++)
			{
				descA_global[i]=0;
				descB_global[i]=0;
			}
			descA_global[1]=-1;
			descB_global[1]=-1;
		}

		// spread matrices
		pdgemr2d_(&n, &n, A_global, &i1, &i1, descA_global, A, &i1, &i1, descA, &context);
		pdgemr2d_(&n, &m, B_global, &i1, &i1, descB_global, B, &i1, &i1, descB, &context);

		// init X and K
		/*
		 * X=Diag(1/A)
		 * K=(A.X)' = X'.A'  ::: factors in reverse order !
		 *
		 * T is the superposition of X and K after having set K[i,i]=0 (i=1..n)
		 *
		 * or, equivalently
		 *
		 * T=(X'.A') after having set A[i,i]=1 (i=1..n)
		 */

		int icol,irow,r,c;

		// distributed init to 0 for mat X (= Tmp1)
		pdlaset_("A", &n, &n, &d0, &d0, Tmp1, &i1, &i1, descTmp1);

		// initialize in parallel the local parts
		//   of X (diagonal elements to the reciprocal of same elements in A)
		//   of K (diagonal elements to "1")
		for (i=1; i<=n; i++)
		{
			// https://info.gwdg.de/wiki/doku.php?id=wiki:hpc:scalapack
			r    = indxg2l_(&i,&nb,&i0,&i0,&nprow);
			irow = indxg2p_(&i,&nb,&i0,&i0,&nprow);
			c    = indxg2l_(&i,&nb,&i0,&i0,&npcol);
			icol = indxg2p_(&i,&nb,&i0,&i0,&npcol);
			if (myrow==irow && mycol==icol)
			{
				Tmp1[c-1+(r-1)*lld]=1/A[c-1+(r-1)*lld];	// X=Diag(1/A)
				A[c-1+(r-1)*lld]=1;						// A[i,i]=1 (i=1..n)
			}
		}

		// X'.A' -> T (= Tmp1'.A' -> Tmp2 )
		// do not transpose operands explicitly because pblas already does, but actual result is T'
		//pdgemm_("N", "N", &n, &n, &n, &d1, Tmp1, &i1, &i1, descTmp1, A, &i1, &i1, descA, &d0, Tmp2, &i1, &i1, descTmp2);
		//pdtran_(&n, &n, &d1, Tmp2, &i1, &i1, descTmp2, &d0, A, &i1, &i1, descA); // transpose result ( (T')'=T -> A )

		// or transpose operands and reverse order to have straight T (in Tmp2)
		pdgemm_("T", "T", &n, &n, &n, &d1, A, &i1, &i1, descA, Tmp1, &i1, &i1, descTmp1, &d0, Tmp2, &i1, &i1, descTmp2);

		pdlaset_("A", &i1, &n, &d1, &d1, J, &i1, &i1, descJ);

		int n_1=n-1;
		double d1_ = -1;

		//pdlacpy_ ("A", &n, &n, Tmp2, &i1, &i1, descTmp2, A, &i1, &i1, descA);

		// a-b*c
		pdger_( &n_1, &n_1, &d1_,
				Tmp2, &i1, &n, descTmp2, &i1,
				Tmp2, &n, &i1, descTmp2, &n,
				Tmp2, &i1, &i1, descTmp2);

		pdgemr2d_(&n, &n, Tmp2, &i1, &i1, descTmp2, A_global, &i1, &i1, descA_global, &context);
		if (mpi_rank==0)
		{
			printf("\n");
			PrintMatrix1D(A_global,n,n);
		}

		// last col in A
		pdgemm_("T", "N", &n, &n, &i1, &d1, J, &i1, &i1, descJ, Tmp2, &n, &i1, descTmp2, &d0, A, &i1, &i1, descA);
		//TODO: size n-1

		// last row in Tmp1
		pdgemm_("T", "T", &n, &n, &i1, &d1, J, &i1, &i1, descJ, Tmp2, &i1, &n, descTmp2, &d0, Tmp1, &i1, &i1, descTmp1);
		//TODO: size n-1

		pdgemr2d_(&n, &n, A, &i1, &i1, descA, A_global, &i1, &i1, descA_global, &context);
		if (mpi_rank==0)
		{
			printf("\n");
			PrintMatrix1D(A_global,n,n);
		}


		// topological formula
		for (i=0; i<nr; i++)
		{
			for (j=0; j<nc; j++)
			{
				A[i*lld+j]=Tmp2[i*lld+j]/(1-Tmp1[i*lld+j]*A[i*lld+j]); // topological
				//A[i*lld+j]=1/(1-Tmp1[i*lld+j]*A[i*lld+j]); // H only
				//A[i*lld+j]=(Tmp2[i*lld+j]-At[i*lld+j]); // a-b*c only
			}
		}
		//TODO: size n*(l-1)

		pdgemr2d_(&n, &n, A, &i1, &i1, descA, A_global, &i1, &i1, descA_global, &context);
		if (mpi_rank==0)
		{
			printf("\n");
			PrintMatrix1D(A_global,n,n);
		}

		// transpose system matrix
		pdtran_(&n, &n, &d1, A, &i1, &i1, descA, &d0, At, &i1, &i1, descAt);

		// linear system equations solver
		result.core_start_time = time(NULL);
		//pdgesv_( &n, &m, At, &i1, &i1, descAt, ipiv, B, &i1, &i1, descB, &info );
		result.core_end_time = time(NULL);
		result.exit_code = info;

		// re-transpose result
		pdtran_(&m, &n, &d1, B, &i1, &i1, descB, &d0, Bt, &i1, &i1, descBt);

		// collect result
		if (mpi_rank==0)
		{
			// Descriptors (global)
			descinit_( descB_global, &m, &n, &i1, &i1, &i0, &i0, &context_global, &m, &info );
		}
		pdgemr2d_(&m, &n, Bt, &i1, &i1, descBt, B_global, &i1, &i1, descB_global, &context);
	}
	else
	{
		A  = NULL;
		At = NULL;
		B  = NULL;
		Bt = NULL;

		Tmp1  = NULL;
		Tmp2 = NULL;

		ipiv = NULL;

		result.core_start_time = time(NULL);
		result.core_end_time = time(NULL);
		result.exit_code = 0;
	}

	// cleanup
	NULLFREE(A);
	NULLFREE(At);
	NULLFREE(B);
	NULLFREE(Bt);

	NULLFREE(Tmp1);
	NULLFREE(Tmp2);

	NULLFREE(ipiv);

	MPI_Barrier(MPI_COMM_WORLD);

	result.total_end_time = time(NULL);

	return result;
}
