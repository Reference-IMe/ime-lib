#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include <mpi.h>

#include <unistd.h>

void SLGEWOPV_calc_last(double** A, double* b, double* x, int n, int rank, int nprocs)
{
	int i,j,l;						// indexes
    int Tcols=2*n;					// total num of cols (X + K)
    int myTcols;					// num of cols per process
    	myTcols=Tcols/nprocs;
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

    double* TlastKc;					// last col of T (K)
    		TlastKc=AllocateVector(n);
    double* TlastKr;					// last row of T (K part)
    		TlastKr=AllocateVector(n);
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
				map[i]= i % nprocs;			// who has the col i
				local[i]=floor(i/nprocs);	// position of the column i(global) in the local matrix
			}
    int*	global;
    		global=malloc(myTcols*sizeof(int));
			for(i=0; i<myTcols; i++)
			{
				global[i]= i * nprocs + rank; // position of the column i(local) in the global matrix
			}

    /*
     * MPI derived types
     */

	MPI_Datatype single_column;
	MPI_Type_vector (n, 1, Tcols, MPI_DOUBLE, & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/nprocs, 1, nprocs, MPI_DOUBLE, & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	MPI_Datatype interleaved_row_resized;
	MPI_Type_create_resized (interleaved_row, 0, 1*sizeof(double), & interleaved_row_resized);
	MPI_Type_commit (& interleaved_row_resized);

	MPI_Datatype multiple_column;
	MPI_Type_vector (n * myTcols, 1, nprocs , MPI_DOUBLE, & multiple_column );
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
	MPI_Bcast (&TlastKc[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast (&TlastKr[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//TODO: combine previous broadcasts in a single one

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
		MPI_Bcast (&TlastKc[0], n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
		MPI_Bcast (&TlastKr[0], n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
		//TODO: substitute Gather with an All-to-All
		//TODO: or combine broadcasts in a single one
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
	DeallocateVector(TlastKc);
	DeallocateVector(TlastKr);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Tlocal,n,CONTIGUOUS);
    if (rank==0)
    {
    	DeallocateMatrix2D(T,n,CONTIGUOUS);
    }
}

/*
 * Keep for reference to understand code changes
 *
 * transition version:
 * from fully allocate matrix to partially allocated
 * - local <-> global indexes defined
 * - loops boundary defined for each node
 *
 */

void SLGEWOPV_calc_last_transition(double** A, double* b, double** T, double* x, int n, double* h, double* hh, int rank, int nprocs)
{
	// indexes
    int i,j,l;

    // num of cols per process
    int mycols;
    mycols=2*n/nprocs;
    int myend=mycols/2;
    int mystart;


    // local storage for a part of the input matrix (continuous columns, not interleaved)
    double** Tlocal;
    Tlocal=AllocateMatrix2D(n, mycols, CONTIGUOUS);

    int* local;
    local=malloc(2*n*sizeof(int));

    // map columns to process
    int* map;
    map=malloc(2*n*sizeof(int));
	for (i=0; i<2*n; i++)
	{
		map[i]= i % nprocs;			// who has the col i
		local[i]=floor(i/nprocs);	// position of the column i(global) in the local matrix
	}

    int* global;
    global=malloc(mycols*sizeof(int));
    for(i=0; i<mycols; i++)
	{
    	global[i]= i * nprocs + rank; // position of the column i(local) in the global matrix
	}

    /*
    if (rank==1)
    {
    	for (i=0; i<2*n; i++)
    	{
		printf("%d\t%d\t%d\n",local[i],i,global[local[i]]);
    	}
    }
    */

    // MPI derived types
	MPI_Datatype single_column;
	MPI_Type_vector (n, 1, 2*n, MPI_DOUBLE, & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/nprocs, 1, nprocs, MPI_DOUBLE, & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	MPI_Datatype interleaved_row_type;
	MPI_Type_create_resized (interleaved_row, 0, 1*sizeof(double), & interleaved_row_type);
	MPI_Type_commit (& interleaved_row_type);

	MPI_Datatype multiple_column;
	MPI_Type_vector (n * mycols, 1, nprocs , MPI_DOUBLE, & multiple_column );
	MPI_Type_commit (& multiple_column);

	MPI_Datatype multiple_column_type;
	MPI_Type_create_resized (multiple_column, 0, 1*sizeof(double), & multiple_column_type);
	MPI_Type_commit (& multiple_column_type);

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
		for (j=0;j<n;j++)
		{
			for (i=0;i<n;i++)
			{
				T[i][j+n]=A[j][i]/A[i][i];
				T[i][j]=0;
			}
			T[j][j]=1/A[j][j];
		}
	}

	// scatter columns to nodes
	MPI_Scatter (&T[0][0], 1, multiple_column_type, &T[0][rank], 1, multiple_column_type, 0, MPI_COMM_WORLD);

	// broadcast of the last col of T (K part)
	MPI_Bcast (&T[0][n*2-1], 1, single_column, 0, MPI_COMM_WORLD);

	// broadcast of the last row of T (K part)
	MPI_Bcast (&T[n-1][n], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//TODO: combine previous broadcasts in a single one

	/*
	 *  calc inhibition sequence
	 */

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		// to avoid IFs: each process loops on its own set of cols, with indirect addressing

		mystart=local[l];
		if (rank<map[l])
		{
			mystart++;
		}
		for (i=mystart; i<=local[n-1]; i++)
		{
			x[global[i]]=x[global[i]]+T[l][global[i]]*b[l];
		}

		myend=local[n+l-1];
		if (rank>map[n+l-1])
		{
			myend--;
		}
		mystart=local[l+1];
		if (rank<map[l+1])
		{
			mystart++;
		}
		for (i=0; i<=l-1; i++)
		{
			// every process
			*h   =1/(1-T[i][n+l]*T[l][n+i]);
			b[i]=b[i]-T[l][n+i]*b[l];

			// processes with diagonal elements not null
			if (rank==map[i])
			{
				T[i][i]=T[i][i]*(*h);
			}

			// process with full not null column (column l)
			if (rank==map[l])
			{
				*hh  =T[i][n+l]*(*h);
				T[i][l]= -T[l][l]*(*hh);
			}

			// all other processes
			for (j=mystart; j<=myend; j++)
				{
						*hh  =T[i][n+l]*(*h);
						T[i][global[j]]=T[i][global[j]]*(*h)-T[l][global[j]]*(*hh);
				}
		}

		// collect chunks of K to "future" last node
		MPI_Gather (&T[l-1][n+rank], 1, interleaved_row_type, &T[l-1][n], 1, interleaved_row_type, map[l-1], MPI_COMM_WORLD);

		//future last node broadcasts last row and col of K
		MPI_Bcast (&T[0][n+l-1], 1, single_column, map[l-1], MPI_COMM_WORLD);
		MPI_Bcast (&T[l-1][n], n, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);

		//TODO: substitute Gather with an All-to-All
		//TODO: or combine broadcasts in a single one
	}

	// last level (l=0)
	for (i=0; i<=n-1; i++)
	{
		if (map[i]==rank)
		{
			x[i]=x[i]+T[0][i]*b[0];
		}
	}

	// collect solution
	MPI_Gather (&x[rank], 1, interleaved_row_type, &x[0], 1, interleaved_row_type, 0, MPI_COMM_WORLD);

	free(local);
	free(global);
	free(map);
	DeallocateMatrix2D(Tlocal,n,CONTIGUOUS);
}
