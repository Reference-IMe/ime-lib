#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include <mpi.h>

#include <unistd.h>

//void SLGEWOPV_calc_last_cs(double** A, double* b, double* x, int n, int rank, int cprocs, int sprocs, double** T, double** Tlocal, double* TlastKc, double* TlastKr, double* h, double* hh)
void SLGEWOPV_calc_last_cs(double** A, double* b, double* x, int n, int rank, int cprocs, int sprocs)
{
	int i,j,l;						// indexes
	int XKcols=2*n;					// num of cols X + K
    int myTcols;					// num of cols per process
    	myTcols=XKcols/cprocs;
    int Scols;
    	Scols=myTcols*sprocs;
    int Tcols=XKcols+Scols;			// num of cols X + K + S
    int myend;						// loop boundaries on local cols
    int mystart;

    int nprocs=cprocs+sprocs;

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
				//T=AllocateMatrix2D(n,Tcols,CONTIGUOUS);
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
    		if (sprocs>0)						// with checksum cols
    		{
				for (i=0; i<Scols; i++)
				{
					map[i]= 0;					// root has the first cols of T (S)
					local[i]=i;					// position of the column i(global) in the local matrix
				}
				for (i=Scols; i<Tcols; i++)
				{
					map[i]= (i-Scols) % cprocs + sprocs;	// who has the other cols i (from rank 1 onwards)
					local[i]=floor((i-Scols)/cprocs);		// position of the column i(global) in the local matrix
				}
    		}
    		else								// without checksum cols
    		{
				for (i=0; i<Tcols; i++)
				{
					map[i]= i % cprocs;			// who has the col i
					local[i]=floor(i/cprocs);	// position of the column i(global) in the local matrix
				}
    		}
    int*	global;
    		global=malloc(myTcols*sizeof(int));
    		if (sprocs>0)						// with checksum cols
    		{
    			if (rank==0)
    			{
					for(i=0; i<myTcols; i++)
					{
						global[i]= i ; 			// root has the checksum cols (in the first positions of the column i(local) in the global matrix)
					}
    			}
    			else
    			{
    				for(i=0; i<myTcols; i++)
					{
						global[i]= myTcols + i * cprocs + rank -1; // position of the column i(local) in the global matrix (-1 because the first non S rank is 1)
					}
    			}
    		}
    		else									// without checksum cols
    		{
    			for(i=0; i<myTcols; i++)
    			{
    				global[i]= i * cprocs + rank; 	// position of the column i(local) in the global matrix
    			}
    		}

    /*
     * MPI derived types
     */

	MPI_Datatype single_column;
	MPI_Type_vector (n, 1, Tcols, MPI_DOUBLE, & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype local_single_column;
	MPI_Type_vector (n, 1, myTcols, MPI_DOUBLE, & local_single_column );
	MPI_Type_commit (& local_single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (myTcols, 1, cprocs, MPI_DOUBLE, & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	MPI_Datatype interleaved_row_resized;
	MPI_Type_create_resized (interleaved_row, 0, 1*sizeof(double), & interleaved_row_resized);
	MPI_Type_commit (& interleaved_row_resized);

	int i_am_calc;
		if (rank==0)
		{
			i_am_calc=0;
		}
		else
		{
			i_am_calc=1;
		}

	MPI_Comm comm_calc;
	MPI_Comm_split(MPI_COMM_WORLD, i_am_calc, rank, &comm_calc);

	int rank_calc;
	MPI_Comm_rank(comm_calc, &rank_calc);

	MPI_Request request = MPI_REQUEST_NULL;

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
				T[i][Scols+j]=0;
			}
			T[j][Scols+j]=1/A[j][j];
			MPI_Isend (&T[0][Scols+j], 1, single_column, map[Scols+j], Scols+j, MPI_COMM_WORLD, &request);

			for (i=0; i<n; i++)
			{
				T[i][Scols+j+n]=A[j][i]/A[i][i];
			}
			MPI_Isend (&T[0][Scols+j+n], 1, single_column, map[Scols+j+n], Scols+j+n, MPI_COMM_WORLD, &request);
		}
		for (j=0; j<Scols; j++)
		{
			for (i=0; i<n; i++)
			{
				T[i][j]=0;
				for (l=0; l<cprocs; l++)
				{
					T[i][j]=T[i][j]+T[i][Scols+j*cprocs+l];
				}
			}
			MPI_Isend (&T[0][j], 1, single_column, map[j], j, MPI_COMM_WORLD, &request);
		}

		// copy data into local buffer before broadcast
		for (i=0; i<n; i++)
		{
			TlastKc[i]=T[i][Scols+n*2-1];
			TlastKr[i]=T[n-1][Scols+n+i];
		}
	}

	// scatter columns to nodes
	// TODO: define new communicator for scattering, to avoid multiple p2p

	// receive
	for (j=0; j<myTcols; j++)
	{
		MPI_Recv (&Tlocal[0][j], 1, local_single_column, 0, global[j], MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// broadcast of the last col and the last row of T (K part)
	MPI_Bcast (&TlastK[0][0], 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/*
	 *  calc inhibition sequence
	 */

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		// update helpers
		// every process
		for (i=0; i<=l-1; i++)
		{
			h[i]   = 1/(1-TlastKc[i]*TlastKr[i]);
			hh[i]  = TlastKc[i]*h[i];
			b[i]   = b[i]-TlastKr[i]*b[l];
		}

		// only to debug
		/*
		for (j=0; j<Tcols; j++)
		{
			if (rank==map[j])
			{
				//printf("%d#%d (%d) sends %d with tag %d\n",l,rank,map[j],j,j);
				MPI_Send (&Tlocal[0][local[j]], 1, local_single_column, 0, j, MPI_COMM_WORLD);
			}
		}
		if (rank==0)
		{
			for (j=0; j<Tcols; j++)
			{
				//printf("%d#%d recvs %d from %d with tag %d\n",l,rank,j,map[j],j);
				MPI_Recv (&T[0][j], 1, single_column, map[j], j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			printf("\n*%d::\n",l);
			PrintMatrix2D(T,n,Tcols);
			printf("lastC:\n");
			PrintVector(TlastKc,n);
			printf("lastR:\n");
			PrintVector(TlastKr,n);
			printf("h:\n");
			PrintVector(h,n);
			printf("\nb:\n");
			PrintVector(b,n);
		}
		*/
		// end debug


		if (rank>=sprocs)
		{
			// update solutions
			// l .. n-1
			mystart=local[Scols+l];
			if (rank<map[Scols+l])
			{
				mystart++;
			}
			for (i=mystart; i<=local[Scols+n-1]; i++)
			{
				x[global[i]-Scols]=x[global[i]-Scols]+Tlocal[l][i]*b[l];
			}

			// update T
			// to avoid IFs: each process loops on its own set of cols, with indirect addressing

			// 0 .. l-1
			// processes with diagonal elements not null
			mystart=local[Scols+0];
			myend=local[Scols+l-1];
			if (rank>map[Scols+l-1])
			{
				myend--;
			}
			for (i=mystart; i<=myend; i++)
			{
				Tlocal[global[i]-Scols][i]=Tlocal[global[i]-Scols][i]*h[global[i]-Scols];
			}

			// l
			// process with full not null column (column l)
			if (rank==map[Scols+l])
			{
				for (i=0; i<=l-1; i++)
				{
					Tlocal[i][local[Scols+l]]= -Tlocal[l][local[Scols+l]]*hh[i];
				}
			}

			// l+1 .. n+l-1
			// all other cases
			mystart=local[Scols+l+1];
			if (rank<map[Scols+l+1])
			{
				mystart++;
			}
			myend=local[Scols+2*n-1];
			if (rank>map[Scols+2*n-1])
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
		}
		else // root (containing S)
		{
			for (j=0; j<myTcols; j++)
			{
				for (i=0; i<=l-1; i++)
				{
					Tlocal[i][j]=Tlocal[i][j]*h[i]-Tlocal[l][j]*hh[i];
				}
			}

		}

		// gather chunks of last row of K to "future" last node
		if (rank>=sprocs)
		{
			MPI_Isend (&Tlocal[l-1][local[Scols+n]], myTcols/2, MPI_DOUBLE, map[Scols+n+l-1], rank, MPI_COMM_WORLD, &request);
		}
		if (rank==map[Scols+n+l-1])
		{
			for (j=sprocs; j<nprocs; j++)
			{
				{
					MPI_Recv (&TlastKr[j-sprocs],  1, interleaved_row_resized, j, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
			}
		}

		//future last node broadcasts last row and col of K
		if (rank==map[Scols+n+l-1])
		{
			// copy data into local buffer before broadcast
			for (i=0; i<n; i++)
			{
				TlastKc[i]=Tlocal[i][local[Scols+n+l-1]];
			}
		}

		// broadcast of the last col and the last row of T (K part)
		MPI_Bcast (&TlastK[0][0], 2*n, MPI_DOUBLE, map[Scols+n+l-1], MPI_COMM_WORLD);
	}

	// only to debug
	/*
	for (j=0; j<Tcols; j++)
	{
		if (rank==map[j])
		{
			//printf("%d#%d (%d) sends %d with tag %d\n",l,rank,map[j],j,j);
			MPI_Send (&Tlocal[0][local[j]], 1, local_single_column, 0, j, MPI_COMM_WORLD);
		}
	}
	if (rank==0)
	{
		for (j=0; j<Tcols; j++)
		{
			//printf("%d#%d recvs %d from %d with tag %d\n",l,rank,j,map[j],j);
			MPI_Recv (&T[0][j], 1, single_column, map[j], j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		printf("*%d::\n",l);
		PrintMatrix2D(T,n,Tcols);
		printf("lastC:\n");
		PrintVector(TlastKc,n);
		printf("lastR:\n");
		PrintVector(TlastKr,n);
		printf("h:\n");
		PrintVector(h,n);
		printf("\nb:\n");
		PrintVector(b,n);
	}
	*/
	// end debug


	// last level (l=0)

	// collect solution
	for (j=Scols; j<n+Scols; j++)
	{
		if (rank==map[j])
		{
			x[j-Scols]=x[j-Scols]+Tlocal[0][local[j]]*b[0];
			MPI_Send (&x[j-Scols], 1, MPI_DOUBLE, 0, j, MPI_COMM_WORLD);
		}
		if (rank==0)
		{
			MPI_Recv (&x[j-Scols],  1, MPI_DOUBLE, map[j], j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}

	// cleanup
	free(local);
	free(global);
	free(map);

	/*
	DeallocateVector(TlastKc);
	DeallocateVector(TlastKr);
	*/
	DeallocateMatrix2D(TlastK,2*n,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Tlocal,n,CONTIGUOUS);
    if (rank==0)
    {
    	DeallocateMatrix2D(T,n,CONTIGUOUS);
    }

}
