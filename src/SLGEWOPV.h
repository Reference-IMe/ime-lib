#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include <mpi.h>

#include <unistd.h>

void SLGEWOPV_calc_last(double** A, double* b, double* x, int n, int rank, int cprocs)
{
	int i,j,l;						// indexes
    int Tcols=2*n;					// total num of cols (X + K)
    int myTcols;					// num of cols per process
    	myTcols=Tcols/cprocs;
    int myend;						// loop boundaries on local cols =myTcols/2;
    int mystart;

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Tlocal;
    		 Tlocal=AllocateMatrix2D(n, myTcols, CONTIGUOUS);

    double** T;
			if (rank==0)
			{
				T=AllocateMatrix2D(n,Tcols,CONTIGUOUS);
			}
			else
			{
				T=Tlocal;				// dummy assignment to avoid segfault (i.e. in  next scatter)
			}

			/*
    double* TlastKc;					// last col of T (K)
    		TlastKc=AllocateVector(n);
    double* TlastKr;					// last row of T (K part)
    		TlastKr=AllocateVector(n);
    		*/
	double** TlastK;
			TlastK=AllocateMatrix2D(2,n, CONTIGUOUS); // last col [0] and row [1] of T (K part)
	double* TlastKc=&TlastK[0][0];
	double* TlastKr=&TlastK[1][0];

    double* h;							// helper vectors
    		h=AllocateVector(n);
    double* hh;
			hh=AllocateVector(n);

	/*
	 * map columns to process
	 */

	int*	local;
    		local=malloc(Tcols*sizeof(int));
    int*	map;
    		map=malloc(Tcols*sizeof(int));
			for (i=0; i<Tcols; i++)
			{
				map[i]= i % cprocs;			// who has the col i
				local[i]=floor(i/cprocs);	// position of the column i(global) in the local matrix
			}
    int*	global;
    		global=malloc(myTcols*sizeof(int));
			for(i=0; i<myTcols; i++)
			{
				global[i]= i * cprocs + rank; // position of the column i(local) in the global matrix
			}

    /*
     * MPI derived types
     */

	MPI_Datatype single_column;
	MPI_Type_vector (n, 1, Tcols, MPI_DOUBLE, & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/cprocs, 1, cprocs, MPI_DOUBLE, & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	MPI_Datatype interleaved_row_resized;
	MPI_Type_create_resized (interleaved_row, 0, 1*sizeof(double), & interleaved_row_resized);
	MPI_Type_commit (& interleaved_row_resized);

	MPI_Datatype multiple_column;
	MPI_Type_vector (n * myTcols, 1, cprocs , MPI_DOUBLE, & multiple_column );
	MPI_Type_commit (& multiple_column);

	MPI_Datatype multiple_column_contiguous;
	MPI_Type_vector (n, myTcols, myTcols , MPI_DOUBLE, & multiple_column_contiguous );
	MPI_Type_commit (& multiple_column_contiguous);

	MPI_Datatype multiple_column_resized;
	MPI_Type_create_resized (multiple_column, 0, 1*sizeof(double), & multiple_column_resized);
	MPI_Type_commit (& multiple_column_resized);


	/*
	 *  init inhibition table
	 */

    MPI_Bcast (&b[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (i=0; i<n; i++)
	{
		x[i]=0.0;
	}

	if (rank==0)
	{
		for (j=0; j<n; j++)
		{
			for (i=0; i<n; i++)
			{
				T[i][j+n]=A[j][i]/A[i][i];
				T[i][j]=0;
			}
			T[j][j]=1/A[j][j];
		}
		// copy data into local buffer before broadcast
		for (i=0; i<n; i++)
		{
			TlastKc[i]=T[i][n*2-1];
			TlastKr[i]=T[n-1][n+i];
		}
	}

	// scatter columns to nodes
	MPI_Scatter (&T[0][0], 1, multiple_column_resized, &Tlocal[0][0], 1, multiple_column_contiguous, 0, MPI_COMM_WORLD);

	// broadcast of the last col and the last row of T (K part)
	/*
	MPI_Bcast (&TlastKc[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast (&TlastKr[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//TODO: combine previous broadcasts in a single one
	*/
	MPI_Bcast (&TlastK[0][0], 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/*
	 *  calc inhibition sequence
	 */

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		// update solutions
		// l .. n-1
		mystart=local[l];
		if (rank<map[l])
		{
			mystart++;
		}
		for (i=mystart; i<=local[n-1]; i++)
		{
			x[global[i]]=x[global[i]]+Tlocal[l][i]*b[l];
		}

		// update helpers
		// every process
		for (i=0; i<=l-1; i++)
		{
			h[i]   = 1/(1-TlastKc[i]*TlastKr[i]);
			hh[i]  = TlastKc[i]*h[i];
			b[i]   = b[i]-TlastKr[i]*b[l];
		}

		// update T
		// to avoid IFs: each process loops on its own set of cols, with indirect addressing

		// 0 .. l-1
		// processes with diagonal elements not null
		mystart=0;
		myend=local[l-1];
		if (rank>map[l-1])
		{
			myend--;
		}
		for (i=mystart; i<=myend; i++)
		{
			Tlocal[global[i]][i]=Tlocal[global[i]][i]*h[global[i]];
		}

		// l
		// process with full not null column (column l)
		if (rank==map[l])
		{
			for (i=0; i<=l-1; i++)
			{
				Tlocal[i][local[l]]= -Tlocal[l][local[l]]*hh[i];
			}
		}

		// l+1 .. n+l-1
		// all other cases
		mystart=local[l+1];
		if (rank<map[l+1])
		{
			mystart++;
		}
		myend=local[n+l-1];
		if (rank>map[n+l-1])
		{
			myend--;
		}
		for (j=mystart; j<=myend; j++)
		{
			for (i=0; i<=l-1; i++)
			{
				Tlocal[i][j]=Tlocal[i][j]*h[i]-Tlocal[l][j]*hh[i];
			}
		}

		// collect chunks of last row of K to "future" last node
		MPI_Gather (&Tlocal[l-1][local[n]], myTcols/2, MPI_DOUBLE, &TlastKr[0], 1, interleaved_row_resized, map[l-1], MPI_COMM_WORLD);

		//future last node broadcasts last row and col of K
		if (rank==map[l-1])
		{
			// copy data into local buffer before broadcast
			for (i=0; i<n; i++)
			{
				TlastKc[i]=Tlocal[i][local[n+l-1]];
			}
		}
		/*
		MPI_Bcast (&TlastKc[0], n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
		MPI_Bcast (&TlastKr[0], n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
		//TODO: substitute Gather with an All-to-All
		//TODO: or combine broadcasts in a single one
		*/
		MPI_Bcast (&TlastK[0][0], 2*n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
	}

	// last level (l=0)
	for (i=0; i<myTcols/2; i++)
	{
		x[global[i]]=x[global[i]]+Tlocal[0][i]*b[0];
	}

	// collect solution
	MPI_Gather (&x[rank], 1, interleaved_row_resized, &x[0], 1, interleaved_row_resized, 0, MPI_COMM_WORLD);

	// cleanup
	free(local);
	free(global);
	free(map);
	/*
	DeallocateVector(TlastKc);
	DeallocateVector(TlastKr);
	*/
	DeallocateMatrix2D(TlastK,2,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Tlocal,n,CONTIGUOUS);
    if (rank==0)
    {
    	DeallocateMatrix2D(T,n,CONTIGUOUS);
    }
}


/*
 * IF removed from init
 * IF removed from calc
 * init with p2p
 * calc with collectives
 *
 * ATTENTION: too many send-recv filling the queue
 * TODO: replace with non-blocking comm.
 *
 */
void SLGEWOPV_calc_sendopt(double** A, double* b, double* s, int n, int rank, int cprocs)
{
	int i,j,l;				// indexes
    int mycols;				// num of cols per process
    	mycols=n/cprocs;
    int myend;				// loop boundaries on local cols =mycols;
    int mystart;

	double** X=A;			// aliases
    double*  F=b;

	double**	K;
				if (rank==0)
				{
					K=AllocateMatrix2D(n,n,CONTIGUOUS);
				}

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */

	double*		H;
				H=AllocateVector(n);

    double**	localK;
    			localK=AllocateMatrix2D(n,mycols,CONTIGUOUS);
    double**	localX;
    			localX=AllocateMatrix2D(n,mycols,CONTIGUOUS);

    double		tmpDiag;					// to hold a diagonal element during init

    /*
    double* 	lastKc;						// last col of K
    			lastKc=AllocateVector(n);
    double*		lastKr;						// last row of K
				lastKr=AllocateVector(n);
	*/
    double** 	lastK;
    			lastK=AllocateMatrix2D(2,n,CONTIGUOUS);
    double* 	lastKc=&lastK[0][0];
    double* 	lastKr=&lastK[1][0];

    /*
	 *  map columns to process
	 */

    int*	local;
    		local=malloc(n*sizeof(int));
	int*	map;
    		map=malloc(n*sizeof(int));
			for (i=0; i<n; i++)
			{
				map[i]= i % cprocs;			// who has the col i
				local[i]=floor(i/cprocs);	// position of the column i(global) in the local matrix

				s[i]=0.0;					// and init solution vector
			}
	int* 	global;
    		global=malloc(n*sizeof(int));
			for(i=0; i<mycols; i++)
			{
				global[i]= i * cprocs + rank; // position of the column i(local) in the global matrix
			}

	/*
	 * MPI derived types
	 */

	MPI_Datatype single_column;
	MPI_Type_vector (n, 1, n, MPI_DOUBLE, & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype local_single_column;
	MPI_Type_vector (n, 1, mycols, MPI_DOUBLE, & local_single_column );
	MPI_Type_commit (& local_single_column);

	MPI_Datatype multiple_column;
	MPI_Type_vector (n * mycols, 1, cprocs , MPI_DOUBLE, & multiple_column );
	MPI_Type_commit (& multiple_column);

	MPI_Datatype multiple_column_resized;
	MPI_Type_create_resized (multiple_column, 0, 1*sizeof(double), & multiple_column_resized);
	MPI_Type_commit (& multiple_column_resized);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/cprocs, 1, cprocs, MPI_DOUBLE, & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	MPI_Datatype interleaved_row_resized;
	MPI_Type_create_resized (interleaved_row, 0, 1*sizeof(double), & interleaved_row_resized);
	MPI_Type_commit (& interleaved_row_resized);


	/*
	 * init inhibition table
	 */

    MPI_Bcast (&b[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//send and recv inside loops, trying to overlap with calculations
	if (rank==0)
	{
		for (j=0; j<n; j++)
		{
			// init col of K
			for (i=0; i<n; i++)
			{
				K[i][j]=A[j][i]/A[i][i];
				// when i==j -> K[i][j]=1
			}

			MPI_Send (&K[0][j], 1, single_column, map[j], j+n, MPI_COMM_WORLD);
		}
		for (j=0; j<n; j++)
		{
			// init col of X
			tmpDiag=1/A[j][j];
			for (i=0; i<n; i++)
			{
				X[i][j]=0.0;
			}
			X[j][j]=tmpDiag;

			MPI_Send (&X[0][j], 1, single_column, map[j], j, MPI_COMM_WORLD);
		}
		// copy data into local buffer before broadcast
		for (i=0; i<n; i++)
		{
			lastKc[i]=K[i][n-1];
			lastKr[i]=K[n-1][i];
		}
	}

	// receive
	for (j=0; j<mycols; j++)
	{
		MPI_Recv (&localX[0][j], 1, local_single_column, 0, global[j], MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv (&localK[0][j], 1, local_single_column, 0, global[j]+n, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// broadcast last column and last row of K
	/*
	MPI_Bcast (&lastKc[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast (&lastKr[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//TODO: combine previous broadcasts in a single one
	*/
	MPI_Bcast (&lastK[0][0], 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/*
	 *  calc inhibition sequence
	 */

	for (l=n-1; l>0; l--)
	{

		for (i=0; i<l; i++)
		{
			H[i]=1/(1-lastKr[i]*lastKc[i]);
			F[i]=F[i]-F[l]*lastKr[i];
		}
		for (j=rank; j<l; j+=cprocs)
		{
			localX[j][local[j]]=H[j]*(localX[j][local[j]]-lastKc[j]*localX[l][local[j]]);
		}
		for (i=0; i<l; i++)
		{
			for (j=n-(cprocs-rank-1)-1; j>=l; j-=cprocs)
			{
				localX[i][local[j]]=H[i]*(localX[i][local[j]]-lastKc[i]*localX[l][local[j]]);
			}
			for (j=rank; j<l; j+=cprocs)
			{
				localK[i][local[j]]=H[i]*(localK[i][local[j]]-lastKc[i]*localK[l][local[j]]);
			}
		}

		// collect chunks of last row of K to "future" last node
		MPI_Gather (&localK[l-1][0], mycols, MPI_DOUBLE, &lastKr[0], 1, interleaved_row_resized, map[l-1], MPI_COMM_WORLD);

		//future last node broadcasts last row and col of K
		if (rank==map[l-1])
		{
			// copy data into local buffer before broadcast
			for (i=0; i<n; i++)
			{
				lastKc[i]=localK[i][local[l-1]];
			}
		}
		/*
		MPI_Bcast (&lastKc[0], n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
		MPI_Bcast (&lastKr[0], n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
		//TODO: combine previous broadcasts in a single one
		*/
		MPI_Bcast (&lastK[0][0], 2*n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
	}

	// calc solution on nodes
	for (j=rank; j<n; j+=cprocs)
	{
		for (l=0; l<n; l++)
		{
			s[j]=F[l]*localX[l][local[j]]+s[j];
		}
	}

	// collect solution
	MPI_Gather (&s[rank], 1, interleaved_row, &s[0], 1, interleaved_row_resized, 0, MPI_COMM_WORLD);

	// cleanup
	free(local);
	free(global);
	free(map);
	free(H);
	DeallocateMatrix2D(localK,n,CONTIGUOUS);
	DeallocateMatrix2D(localX,n,CONTIGUOUS);
	/*
	DeallocateVector(lastKr);
	DeallocateVector(lastKc);
	*/
	DeallocateMatrix2D(lastK,2,CONTIGUOUS);
	if (rank==0)
	{
		DeallocateMatrix2D(K,n,CONTIGUOUS);
	}
}
