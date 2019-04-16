#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include <mpi.h>

#include <unistd.h>

void SLGEWOPV_calc_last_mcs(double** A, double* b, double* x, int n, int rank, int cprocs, int sprocs)
{
	int i,j,l,p;						// indexes
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
				for (i=XKcols; i<Tcols; i++)
				{
					map[i]= cprocs + ((i-XKcols) % sprocs);				// n+1th has the first cols of T (S)
					local[i]= (i-XKcols) % sprocs;				// position of the column i(global) in the local matrix
				}
				for (i=0; i<XKcols; i++)
				{
					map[i]= i % cprocs;			// who has the other cols i (from rank 1 onwards)
					local[i]=floor(i/cprocs);	// position of the column i(global) in the local matrix
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
    			if (rank>=cprocs)
    			{
					for(i=0; i<myTcols; i++)
					{
						global[i]= XKcols + i + (rank-cprocs) * myTcols; 	// n+1th has the checksum cols (in the last positions of the column i(local) in the global matrix)
					}
    			}
    			else
    			{
    				for(i=0; i<myTcols; i++)
					{
						global[i]= i * cprocs + rank; // position of the column i(local) in the global matrix
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

	MPI_Datatype single_column_resized;
	MPI_Type_create_resized (single_column, 0, 1*sizeof(double), & single_column_resized);
	MPI_Type_commit (& single_column_resized);

	MPI_Datatype local_single_column;
	MPI_Type_vector (n, 1, myTcols, MPI_DOUBLE, & local_single_column );
	MPI_Type_commit (& local_single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/cprocs, 1, cprocs, MPI_DOUBLE, & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	MPI_Datatype interleaved_row_resized;
	MPI_Type_create_resized (interleaved_row, 0, 1*sizeof(double), & interleaved_row_resized);
	MPI_Type_commit (& interleaved_row_resized);

	int i_am_calc;
		if (rank>=cprocs)
		{
			i_am_calc=0;
		}
		else
		{
			i_am_calc=1;
		}

	MPI_Comm comm_calc;
	int rank_calc;
	if (sprocs>0)
	{
		MPI_Comm_split(MPI_COMM_WORLD, i_am_calc, rank, &comm_calc);
		MPI_Comm_rank(comm_calc, &rank_calc);
	}
	else
	{
		comm_calc=MPI_COMM_WORLD;
		rank_calc=rank;
	}

	/*
	 *  init inhibition table
	 */

    MPI_Bcast (&b[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (i_am_calc)
    {
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
					T[i][j]=0;
					T[i][j+n]=A[j][i]/A[i][i];
				}
				T[j][j]=1/A[j][j];
			}
			for (p=0; p<sprocs; p++)
			{
				for (j=0; j<myTcols; j++)
				{
					for (i=0; i<n; i++)
					{
						T[i][XKcols+p*myTcols+j]=0;
						for (l=0; l<cprocs; l++)
						{
							T[i][XKcols+p*myTcols+j]=T[i][XKcols+p*myTcols+j]+T[i][j*cprocs+l];
						}
					}
				}
			}

			// copy data into local buffer preparing for the next broadcast
			for (i=0; i<n; i++)
			{
				TlastKc[i]=T[i][n*2-1];
				TlastKr[i]=T[n-1][n+i];
			}
			for (j=0; j<sprocs; j++)
			{
				// send checksumming cols
				MPI_Send (&T[0][XKcols+j*myTcols], myTcols, single_column_resized, cprocs+j, cprocs+j, MPI_COMM_WORLD);
			}
		}
		// scatter other columns to nodes
		for (j=0; j<myTcols; j++)
		{
			MPI_Scatter (&T[0][j*cprocs], 1, single_column_resized, &Tlocal[0][j], 1, local_single_column, 0, comm_calc);
		}
    }
    else // checksum node
    {
    	MPI_Recv (&Tlocal[0][0], n*myTcols, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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


		if (rank<cprocs)
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

			// update T
			// to avoid IFs: each process loops on its own set of cols, with indirect addressing

			// 0 .. l-1
			// processes with diagonal elements not null
			mystart=local[0];
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

			myend=local[2*n-1];

			if (rank>map[2*n-1])
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

			// gather chunks of last row of K to "future" last node
			MPI_Gather (&Tlocal[l-1][local[n]], myTcols/2, MPI_DOUBLE, &TlastKr[0], 1, interleaved_row_resized, map[l-1], comm_calc);
		}
		else // node containing S
		{
			for (j=0; j<myTcols; j++)
			{
				for (i=0; i<=l-1; i++)
				{
					Tlocal[i][j]=Tlocal[i][j]*h[i]-Tlocal[l][j]*hh[i];
				}
			}

		}

		//future last node broadcasts last row and col of K
		if (rank==map[n+l-1])
		{
			// copy data into local buffer before broadcast
			for (i=0; i<n; i++)
			{
				TlastKc[i]=Tlocal[i][local[n+l-1]];
			}
		}

		// broadcast of the last col and the last row of T (K part)
		MPI_Bcast (&TlastK[0][0], 2*n, MPI_DOUBLE, map[n+l-1], MPI_COMM_WORLD);
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

	if (i_am_calc)
	{
		for (i=0; i<myTcols/2; i++)
		{
			x[global[i]]=x[global[i]]+Tlocal[0][i]*b[0];
		}
		// collect solution
		MPI_Gather (&x[rank], 1, interleaved_row_resized, &x[0], 1, interleaved_row_resized, 0, comm_calc);
	}


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
