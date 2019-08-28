#include <mpi.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "DGEZR.h"
#include "DGEIT_Wx.h"

/*
 *	solve (SV) system with general (GE) matrix A of doubles (D)
 *	of order n and with m r.h.s in matrix bb and solutions in xx
 *	with wide overwrite (WO) memory model
 *	parallelized in interleaved columns (pvi) over cprocs calculating processors
 *	without optimized initialization
 *
 */

void pviDGESV_WO_swaploop(int n, double** A, int m, double** bb, double** xx, int rank, int cprocs)
{
	int i,j,l;						// indexes
    int Tcols=2*n;					// total num of cols (X + K)
    int myTcols;					// num of cols per process
    	myTcols=Tcols/cprocs;
    int myend;						// loop boundaries on local cols =myTcols/2;
    int mystart;
    int rhs;

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
				T=Tlocal;				// dummy assignment to avoid segfault (i.e. in next scatter)
			}

	double** TlastK;
			TlastK=AllocateMatrix2D(2,n, CONTIGUOUS);	// last col [0] and row [1] of T (K part)
	double* TlastKc=&TlastK[0][0];						// alias for last col
	double* TlastKr=&TlastK[1][0];						// alias for last row

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

	MPI_Datatype multiple_row;
	MPI_Type_vector (n/cprocs, m, m*cprocs, MPI_DOUBLE, & multiple_row );
	MPI_Type_commit (& multiple_row);

	MPI_Datatype multiple_row_resized;
	MPI_Type_create_resized (multiple_row, 0, m*sizeof(double), & multiple_row_resized);
	MPI_Type_commit (& multiple_row_resized);

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

    MPI_Bcast (&bb[0][0], n*m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    DGEZR(xx, n, m);			// init (zero) solution vectors

	if (rank==0)
	{
		DGEIT_W(A, T, n);		// init inhibition table

		for (i=0; i<n; i++)		// copy data into local buffer before broadcast
		{
			TlastKc[i]=T[i][n*2-1];
			TlastKr[i]=T[n-1][n+i];
		}
	}

	MPI_Scatter (&T[0][0], 1, multiple_column_resized, &Tlocal[0][0], 1, multiple_column_contiguous, 0, MPI_COMM_WORLD);	// scatter columns to nodes
	MPI_Bcast (&TlastK[0][0], 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);															// broadcast of the last col and the last row of T (K part)

	/*
	 *  calc inhibition sequence
	 */

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		// ALL procs
		// update solutions
		// l .. n-1
		mystart=local[l];
		if (rank<map[l])
		{
			mystart++;
		}
		for (i=mystart; i<=local[n-1]; i++)
		{
			for(rhs=0;rhs<m;rhs++)
			{
				xx[global[i]][rhs]=xx[global[i]][rhs]+Tlocal[l][i]*bb[l][rhs];
			}
		}

		// ALL procs
		// update helpers
		for (i=0; i<=l-1; i++)
		{
			h[i]   = 1/(1-TlastKc[i]*TlastKr[i]);
			hh[i]  = TlastKc[i]*h[i];
			for(rhs=0;rhs<m;rhs++)
			{
				bb[i][rhs] = bb[i][rhs]-TlastKr[i]*bb[l][rhs];
			}
		}

		// update T
		// to avoid IFs: each process loops on its own set of cols, with indirect addressing

		// 0 .. l-1
		// ALL procs
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
		// ONLY one proc
		// process with full not null column (column l)
		if (rank==map[l])
		{
			for (i=0; i<=l-1; i++)
			{
				Tlocal[i][local[l]]= -Tlocal[l][local[l]]*hh[i];
			}
		}// TODO: incorporate with next block to form l .. n+l-1

		// l+1 .. n+l-1
		// ALL procs
		// all other cases
		mystart=local[l+1]; // TODO: incorporate l
		if (rank<map[l+1])
		{
			mystart++;
		}
		myend=local[n+l-1];
		if (rank>map[n+l-1])
		{
			myend--;
		}
		/*
		for (j=mystart; j<=myend; j++) // TODO: swap loops
		{
			for (i=0; i<=l-1; i++)
			{
				Tlocal[i][j]=Tlocal[i][j]*h[i]-Tlocal[l][j]*hh[i];
			}
		}
		*/
		for (i=0; i<=l-1; i++)
		{
			for (j=mystart; j<=myend; j++)
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
		//TODO: or combine broadcasts in a single one -> DONE
		*/
		MPI_Bcast (&TlastK[0][0], 2*n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
	}

	// last level (l=0)
	for (i=0; i<myTcols/2; i++)
	{
		for(rhs=0;rhs<m;rhs++)
		{
			xx[global[i]][rhs]=xx[global[i]][rhs]+Tlocal[0][i]*bb[0][rhs];
		}
	}

	// collect solution
	// MPI_IN_PLACE required for MPICH based versions
	if (rank==0)
	{
		MPI_Gather (MPI_IN_PLACE, 1, multiple_row_resized, &xx[0][0], 1, multiple_row_resized, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Gather (&xx[rank][0], 1, multiple_row_resized, &xx[0][0], 1, multiple_row_resized, 0, MPI_COMM_WORLD);
	}

	// cleanup
	free(local);
	free(global);
	free(map);

	DeallocateMatrix2D(TlastK,2,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Tlocal,n,CONTIGUOUS);
    if (rank==0)
    {
    	DeallocateMatrix2D(T,n,CONTIGUOUS);
    }
}
