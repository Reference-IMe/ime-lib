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
	//int *ipiv;

	// matrix
	int nr, nc;
	double *A;
	//double *At;
	double *C;
	double *T;
	double *J;
	double *diag;
	double *lcol;

	int ncrhs, nrrhs;
	double *B;
	//int ncrhst, nrrhst;
	//double *Bt;

	int descA_global[9];
	int descB_global[9];
	int descA[9];
	//int descAt[9];
	int descC[9];
	int descT[9];
	int descJ[9];
	int descB[9];
	//int descBt[9];

	int lld;
	//int lldt;

	int ncj, nrj;

	result.core_start_time = time(NULL);

	if (mpi_rank < cprocs)
	{
		// Computation of local matrix size
		nc = numroc_( &n, &nb, &mycol, &i0, &npcol );
		nr = numroc_( &n, &nb, &myrow, &i0, &nprow );
		lld = MAX( 1 , nr );
		A  = malloc(nr*nc*sizeof(double));
		//At = malloc(nr*nc*sizeof(double));
		C = malloc(nr*nc*sizeof(double));
		T = malloc(nr*nc*sizeof(double));
		diag = malloc(nr*sizeof(double));
		lcol = malloc(nr*sizeof(double));

		ncj = numroc_( &n, &nb, &mycol, &i0, &npcol );
		nrj = numroc_( &i1,  &nb, &myrow, &i0, &nprow );
		J = malloc(ncj*nrj*sizeof(double));

		ncrhs = numroc_( &m, &nb, &mycol, &i0, &npcol );
		nrrhs = numroc_( &n, &nb, &myrow, &i0, &nprow );
		B = malloc(nrrhs*ncrhs*sizeof(double));

		//ncrhst = numroc_( &n, &nb, &mycol, &i0, &npcol );
		//nrrhst = numroc_( &m, &nb, &myrow, &i0, &nprow );
		//lldt = MAX( 1 , nrrhst );
		//Bt = malloc(nrrhst*ncrhst*sizeof(double));

		//ipiv = malloc((lld+nb)*sizeof(int));

		// Descriptors (local)
		descinit_( descA, &n, &n, &nb, &nb, &i0, &i0, &context, &lld, &info );
		//descinit_( descAt, &n, &n, &nb, &nb, &i0, &i0, &context, &lld, &info );

		descinit_( descC, &n, &n, &nb, &nb, &i0, &i0, &context, &lld, &info );
		descinit_( descT, &n, &n, &nb, &nb, &i0, &i0, &context, &lld, &info );
		descinit_( descJ, &i1, &n, &nb, &nb, &i0, &i0, &context, &i1, &info );

		descinit_( descB, &n, &m, &nb, &nb, &i0, &i0, &context, &lld, &info );
		//descinit_( descBt, &m, &n, &nb, &nb, &i0, &i0, &context, &lldt, &info );

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

		// distributed init to 0 for mat X (= C)
		pdlaset_("A", &n, &n, &d0, &d0, C, &i1, &i1, descC);

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
				C[c-1+(r-1)*lld]=1/A[c-1+(r-1)*lld];	// X=Diag(1/A)
				A[c-1+(r-1)*lld]=1;						// A[i,i]=1 (i=1..n)
			}
		}

		// X'.A' -> T (= C'.A' -> T )
		// do not transpose operands explicitly because pblas already does, but actual result is T'
		//pdgemm_("N", "N", &n, &n, &n, &d1, C, &i1, &i1, descC, A, &i1, &i1, descA, &d0, T, &i1, &i1, descT);
		//pdtran_(&n, &n, &d1, T, &i1, &i1, descT, &d0, A, &i1, &i1, descA); // transpose result ( (T')'=T -> A )

		// or transpose operands and reverse order to have straight T (in Tmp2)
		pdgemm_("T", "T", &n, &n, &n, &d1, A, &i1, &i1, descA, C, &i1, &i1, descC, &d0, T, &i1, &i1, descT);

		/*
		pdgemr2d_(&n, &n, T, &i1, &i1, descT, A_global, &i1, &i1, descA_global, &context);
		if (mpi_rank==0)
		{
			printf("\n[%d]\n",n);
			printf("(T)\n");
			PrintMatrix1D(A_global,n,n);
		}
		*/

		pdlaset_("A", &i1, &n, &d1, &d1, J, &i1, &i1, descJ);

		double d1_ = -1;
		int l;
		int l_1;

		for (l=n; l>1; l--)
		{
			l_1=l-1;
			/*
			// a-b*c 1..l-1
			pdger_( &l_1, &l_1, &d1_,
					Tmp2, &i1, &n, descTmp2, &i1,
					Tmp2, &n, &i1, descTmp2, &n,
					Tmp2, &i1, &i1, descTmp2);

			// a-b*c
			pdger_( &n_l, &l_1, &d1_,
					Tmp2, &lp1, &n, descTmp2, &i1,
					Tmp2, &n, &i1, descTmp2, &n,
					Tmp2, &lp1, &i1, descTmp2);
			*/

			// last col in A
			pdgemm_("T", "N", &n, &n, &i1, &d1, J, &i1, &i1, descJ, T, &l, &i1, descT, &d0, A, &i1, &i1, descA);

			/*
			pdgemr2d_(&n, &n, A, &i1, &i1, descA, A_global, &i1, &i1, descA_global, &context);
			if (mpi_rank==0)
			{
				printf("\n[%d]\n",l-1);
				printf("(col)\n");
				PrintMatrix1D(A_global,n,n);
			}
			*/

			// last row in Tmp1
			pdgemm_("T", "T", &n, &n, &i1, &d1, J, &i1, &i1, descJ, T, &i1, &l, descT, &d0, C, &i1, &i1, descC);

			/*
			pdgemr2d_(&n, &n, C, &i1, &i1, descC, A_global, &i1, &i1, descA_global, &context);
			if (mpi_rank==0)
			{
				printf("(row)\n");
				PrintMatrix1D(A_global,n,n);
			}
			*/

			// save elements that will be overwritten
			//
			// !! swapped indices !! transposition
			//
			for (i=1; i<=l-1; i++) // i should be rows, but treated as cols
			{
				// https://info.gwdg.de/wiki/doku.php?id=wiki:hpc:scalapack
				//r    = indxg2l_(&i,&nb,&i0,&i0,&nprow);
				//irow = indxg2p_(&i,&nb,&i0,&i0,&nprow);
				c    = indxg2l_(&i,&nb,&i0,&i0,&npcol);
				icol = indxg2p_(&i,&nb,&i0,&i0,&npcol);
				if (mycol==icol)
				{
					// diagonal
					r    = indxg2l_(&i,&nb,&i0,&i0,&nprow);
					irow = indxg2p_(&i,&nb,&i0,&i0,&nprow);
					if (myrow==irow)
					{
						diag[r-1]=T[c-1+(r-1)*lld];
					}
					// l-th column, treated as row
					r    = indxg2l_(&l,&nb,&i0,&i0,&nprow);
					irow = indxg2p_(&l,&nb,&i0,&i0,&nprow);
					if (myrow==irow)
					{
						lcol[c-1]=T[r-1+(c-1)*lld];
					}
				}
			}

			// a-b*c
			pdger_( &n, &l_1, &d1_,
					T, &i1, &l, descT, &i1,
					T, &l, &i1, descT, &n,
					T, &i1, &i1, descT);

			/*
			pdgemr2d_(&n, &n, T, &i1, &i1, descT, A_global, &i1, &i1, descA_global, &context);
			if (mpi_rank==0)
			{
				printf("(-)\n");
				PrintMatrix1D(A_global,l-1,n);
			}
			*/

			// save elements that will be overwritten
			//
			// !! swapped indices !! transposition
			//
			for (i=1; i<=l-1; i++) // "i" should be rows, but treated as cols
			{
				// https://info.gwdg.de/wiki/doku.php?id=wiki:hpc:scalapack
				//r    = indxg2l_(&i,&nb,&i0,&i0,&nprow);
				//irow = indxg2p_(&i,&nb,&i0,&i0,&nprow);
				c    = indxg2l_(&i,&nb,&i0,&i0,&npcol);
				icol = indxg2p_(&i,&nb,&i0,&i0,&npcol);
				if (mycol==icol)
				{
					// diagonal
					r    = indxg2l_(&i,&nb,&i0,&i0,&nprow);
					irow = indxg2p_(&i,&nb,&i0,&i0,&nprow);
					if (myrow==irow)
					{
						T[c-1+(r-1)*lld]=diag[r-1];
					}
					// l-th column, treated as row
					r    = indxg2l_(&l,&nb,&i0,&i0,&nprow);
					irow = indxg2p_(&l,&nb,&i0,&i0,&nprow);
					if (myrow==irow)
					{
						T[r-1+(c-1)*lld]=T[(r-1)+(c-1)*lld]-lcol[c-1];
					}
				}
			}

			/*
			pdgemr2d_(&n, &n, T, &i1, &i1, descT, A_global, &i1, &i1, descA_global, &context);
			if (mpi_rank==0)
			{
				printf("(K pre H)\n");
				PrintMatrix1D(A_global,n,n);
			}
			*/

			// topological formula
			// (l-1)-th row, treated as column
			c    = indxg2l_(&l_1,&nb,&i0,&i0,&npcol);
			icol = indxg2p_(&l_1,&nb,&i0,&i0,&npcol); // processor row holding the (l-1)-th matrix row

			if (mycol<icol)			// blocks above, updated in full
			{
				for (i=0; i<nr; i++)
				{
					for (j=0; j<nc; j++)
					{
						//T[j*lld+i]=T[j*lld+i]/(1-(C[j*lld+i]*A[j*lld+i])); // topological
						T[i*lld+j]=T[i*lld+j]/(1-(C[i*lld+j]*A[i*lld+j])); // topological
					}
				}
			}
			else if (mycol==icol)	// blocks containing that row, updated above that row only
			{
				for (i=0; i<c; i++)	// rows above
				{
					for (j=0; j<nc; j++)
					{
						T[i*lld+j]=T[i*lld+j]/(1-(C[i*lld+j]*A[i*lld+j])); // topological
					}
				}
			}
			// else // (mycol>icol) // blocks below
			{
				//do nothing
			}

			pdgemr2d_(&n, &n, T, &i1, &i1, descT, A_global, &i1, &i1, descA_global, &context);
			if (mpi_rank==0)
			{
				printf("T(%d)\n",l-1);
				PrintMatrix1D(A_global,n,n);
			}
		}

		result.core_end_time = time(NULL);
		result.exit_code = 0;

		// re-transpose result
		//pdtran_(&m, &n, &d1, B, &i1, &i1, descB, &d0, Bt, &i1, &i1, descBt);

		/*
		// collect result
		if (mpi_rank==0)
		{
			// Descriptors (global)
			descinit_( descB_global, &m, &n, &i1, &i1, &i0, &i0, &context_global, &m, &info );
		}
		pdgemr2d_(&m, &n, Bt, &i1, &i1, descBt, B_global, &i1, &i1, descB_global, &context);
		*/
	}
	else
	{
		A  = NULL;
		//At = NULL;
		B  = NULL;
		//Bt = NULL;

		J = NULL;
		C = NULL;
		T = NULL;

		//ipiv = NULL;

		result.core_start_time = time(NULL);
		result.core_end_time = time(NULL);
		result.exit_code = 0;
	}

	// cleanup
	NULLFREE(A);
	//NULLFREE(At);
	NULLFREE(B);
	//NULLFREE(Bt);

	NULLFREE(J);
	NULLFREE(C);
	NULLFREE(T);

	//NULLFREE(ipiv);

	MPI_Barrier(MPI_COMM_WORLD);

	result.total_end_time = time(NULL);

	return result;
}
