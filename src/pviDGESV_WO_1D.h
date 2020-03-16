#include <mpi.h>
#include <time.h>
#include "helpers/info.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include "DGEZR.h"
#include "pDGEIT_WX_1D.h"

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
result_info pviDGESV_WO_1D(int bf, int n, double** A, int m, double** bb, double** xx, MPI_Comm comm)
{
	result_info result;

	result.total_start_time = time(NULL);

    int rank, cprocs; //
    MPI_Comm_rank(comm, &rank);		//get current process id
    MPI_Comm_size(comm, &cprocs);	// get number of processes

	int i,j,l;						// indexes
    int mycols;					// num of T cols per process
    	mycols=n/cprocs;
    int myxxrows=n/cprocs;
    int myKcols;
    	myKcols=n/cprocs;
	int myXcols;
		myXcols=n/cprocs;
    //int myend;						// loop boundaries on local cols =myTcols/2;
    int mystart;
    int rhs;

    //int avoidif;					// for boolean --> int conversion

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Xlocal;
    		 Xlocal=AllocateMatrix2D(n, myXcols, CONTIGUOUS);
	double** Klocal;
			 Klocal=AllocateMatrix2D(n, myKcols, CONTIGUOUS);
	double** lastK;
			 lastK=AllocateMatrix2D(2*bf,n, CONTIGUOUS);// last rows [0 - (bf-1)] and cols [ bf - (2bf -1)] of K
	double** lastKr;
				lastKr=malloc(bf*sizeof(double*));
				for(i=0;i<bf;i++)
				{
					lastKr[i]=lastK[i];						// alias for last row
				}
	double** lastKc;
				lastKc=malloc(bf*sizeof(double*));
				for(i=0;i<bf;i++)
				{
					lastKc[i]=lastK[bf+i];						// alias for last col
				}

    double* h;							// helper vectors
    		h=AllocateVector(n);
    double* hh;
			hh=AllocateVector(n);

	/*
	 * map columns to process
	 */
	int*	local;
    		local=malloc(n*sizeof(int));
    int*	map;
    		map=malloc(n*sizeof(int));
			for (i=0; i<n; i++)
			{
				map[i]= ((int)floor(i/bf)) % cprocs;			// who has the col i
				local[i]=(int)floor(i/(bf*cprocs))*bf + i % bf;	// position of the column i(global) in the local matrix
			}
    int*	global;
    		global=malloc(mycols*sizeof(int));
			for (i=0; i<mycols; i++)
			{
				global[i]= i % bf + (int)floor(i/bf)*cprocs*bf + rank*bf; // position of the column i(local) in the global matrix
			}



	MPI_Status  mpi_status;
	MPI_Request mpi_request = MPI_REQUEST_NULL;


    /*
     * MPI derived types
     */

	/*
	 * option 1: derived data type, for interleaving while for gathering
	 *           non working for some values of n and np, why? extent of derived MPI data type?
	 */

	MPI_Datatype lastKr_chunks;
	MPI_Type_vector (myKcols/bf, bf, bf*cprocs, MPI_DOUBLE, & lastKr_chunks );
	MPI_Type_commit (& lastKr_chunks);

	MPI_Datatype multiple_lastKr_chunks;
	MPI_Type_vector (myKcols, bf, bf*cprocs, MPI_DOUBLE, & multiple_lastKr_chunks );
	MPI_Type_commit (& multiple_lastKr_chunks);

	MPI_Datatype lastKr_chunks_resized;
	MPI_Type_create_resized (lastKr_chunks, 0, bf*sizeof(double), & lastKr_chunks_resized);
	MPI_Type_commit (& lastKr_chunks_resized);

	MPI_Datatype multiple_lastKr_chunks_resized;
	MPI_Type_create_resized (multiple_lastKr_chunks, 0, bf*sizeof(double), & multiple_lastKr_chunks_resized);
	MPI_Type_commit (& multiple_lastKr_chunks_resized);


	// rows of xx to be extracted
	MPI_Datatype xx_rows_interleaved;
	MPI_Type_vector (myxxrows/bf, m*bf, bf*m*cprocs, MPI_DOUBLE, & xx_rows_interleaved );
	MPI_Type_commit (& xx_rows_interleaved);

	// rows of xx to be extracted, properly resized for gathering
	MPI_Datatype xx_rows_interleaved_resized;
	MPI_Type_create_resized (xx_rows_interleaved, 0, bf*m*sizeof(double), & xx_rows_interleaved_resized);
	MPI_Type_commit (& xx_rows_interleaved_resized);


    /*
	 *  init inhibition table
	 */
	DGEZR(xx, n, m);															// init (zero) solution vectors
	pDGEIT_WX_1D(A, Xlocal, Klocal, lastK, n, bf, comm, rank, cprocs, map, global, local);	// init inhibition table
    MPI_Bcast (&bb[0][0], n*m, MPI_DOUBLE, 0, comm);							// send all r.h.s to all procs

	/*
	 *  calc inhibition sequence
	 */
	result.core_start_time = time(NULL);

	int myKend;
	myKend=myKcols-1;

	int myXmid;
	myXmid=myXcols-1;

	int myxxstart;
	myxxstart=myXcols-1;

	int bfi=0;

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		if (rank==map[l])
		{
			myKend--;
			myXmid--;
			myxxstart--;
		}
		// ALL procs
		// update solutions
		// l .. n-1
		//avoidif=(rank<map[l]);
		//mystart = local[l] + avoidif;
		//for (i=mystart; i<=local[n-1]; i++)
		for (i=myxxstart; i<=local[n-1]; i++)
		{
			for (rhs=0;rhs<m;rhs++)
			{
				xx[global[i]][rhs]=xx[global[i]][rhs]+Xlocal[l][i]*bb[l][rhs];
			}
		}

		// MPI_Wait(&mpi_request, &mpi_status);

		// ALL procs
		// update helpers
		for (i=0; i<=l-1; i++)
		{
			h[i]   = 1/(1-lastKc[bf-1-bfi][i]*lastKr[bf-1-bfi][i]);
			hh[i]  = lastKc[bf-1-bfi][i]*h[i];
			for (rhs=0;rhs<m;rhs++)
			{
				bb[i][rhs] = bb[i][rhs]-lastKr[bf-1-bfi][i]*bb[l][rhs];
			}
		}

    	MPI_Barrier(comm);
		if (rank==0)
		{
			printf("\nlevel %d\n",l);
			if (bfi==0)
			{
				printf("- lastK received\n");
			}
			else
			{
				printf("- lastK calculated from %d\n",l+1);
			}
			fflush(stdout);
		}
	    for (i=0;i<cprocs;i++)
	    {
	    	MPI_Barrier(comm);
	    	if(rank==i)
	    	{
				printf("\nrank %d(%d):\n",rank,l);
				printf("last R\n");
				PrintMatrix2D(lastKr, bf, n);
				printf("last C\n");
				PrintMatrix2D(lastKc, bf, n);
				printf("h(%d)\n",l);
				PrintVector(h, n);
				printf("\n");
				fflush(stdout);
	    	}
	    }
    	MPI_Barrier(comm);

	    //////////////// update T
		// to avoid IFs: each process loops on its own set of cols, with indirect addressing

		//////// X
		// 0 .. l-1
		// ALL procs
		// processes with diagonal elements not null
		mystart=0;
		for (i=mystart; i<=myXmid; i++)
		{
			Xlocal[global[i]][i]=Xlocal[global[i]][i]*h[global[i]];
		}

		// l .. n-1
		// ALL procs
		for (i=0; i<=l-1; i++)
		{
			for (j=myXmid+1; j<=myXcols-1; j++)
			{
				Xlocal[i][j]=Xlocal[i][j]*h[i] - Xlocal[l][j]*hh[i];
			}
		}

		//////// K
		// 0 .. l-1
		// ALL procs

		mystart=0;
		for (i=0; i<=l-1; i++)
		{
			for (j=mystart; j<=myKend; j++)
			{
				Klocal[i][j]=Klocal[i][j]*h[i] - Klocal[l][j]*hh[i];
			}
		}

		if (bfi<bf-1)
		{
			bfi++;
		    for (i=0;i<cprocs;i++)
		    {
		    	MPI_Barrier(comm);
		    	if(rank==i)
		    	{
					printf("\nrank %d(%d):\n",rank,l);
					printf("pre C\n");
					PrintMatrix2D(lastKc, bf, n);
					printf("\n");
					fflush(stdout);
		    	}
		    }
			for (j=0; j<=l-1; j++)
			{
				//for (i=bf-1;i>=bfi;i--)
				for (i=0;i<bf-bfi;i++)
				{
				//lastKc[bfi][j]=lastKc[bfi][j]*h[j]   - lastKr[bfi-1][l-1]*hh[j];
				//lastKr[bfi][j]=lastKr[bfi][j]*h[l-1] - lastKr[bfi-1][j]*hh[l-1];
				//TODO better indexing technique
				//lastKc[i][j]=lastKc[i][j]*h[j]   - lastKr[bfi-1][l-i+(bfi-1)]*hh[j];
				//lastKr[i][j]=lastKr[i][j]*h[l-i+(bfi-1)] - lastKr[bfi-1][j]*hh[l-i+(bfi-1)];
				lastKc[i][j]=lastKc[i][j]*h[j]   - lastKr[bf-bfi][l-bf+bfi+i]*hh[j];
				lastKr[i][j]=lastKr[i][j]*h[l-bf+bfi+i] - lastKr[bf-bfi][j]*hh[l-bf+bfi+i];
				}
			}
		    for (i=0;i<cprocs;i++)
		    {
		    	MPI_Barrier(comm);
		    	if(rank==i)
		    	{
					printf("\nrank %d(%d):\n",rank,l);
					printf("post C\n");
					PrintMatrix2D(lastKc, bf, n);
					printf("\n");
					fflush(stdout);
		    	}
		    }
		}
		else
		{
			bfi=0;
			{

				// collect chunks of last row of K to "future" last node
				//MPI_Igather (&Klocal[l-1][local[0]], myKcols, MPI_DOUBLE, &lastKr[0], 1, lastKr_chunks_resized, map[l-1], comm, &mpi_request);
				//TODO group in a single gather (after better indexing, see before)
				for (i=0;i<bf;i++)
				{
					MPI_Gather (&Klocal[l-1-i][local[0]], myKcols, MPI_DOUBLE, &lastKr[bf-1-i][0], 1, lastKr_chunks_resized, map[l-bf], comm);
					//MPI_Gather (&Klocal[l-2][local[0]], myKcols, MPI_DOUBLE, &lastKr[1][0], 1, lastKr_chunks_resized, map[l-bf], comm);
				}
				//MPI_Gather (&Klocal[l-bf][local[0]], bf*myKcols, MPI_DOUBLE, &lastKr[0][0], 1, multiple_lastKr_chunks_resized, map[l-bf], comm);

				//future last node broadcasts last rows and cols of K
				if (rank==map[l-bf])
				{
					// copy data into local buffer before broadcast
					for (i=0; i<=l-1; i++)
					{
						for(j=0;j<bf;j++)
						{
							lastKc[j][i]=Klocal[i][local[l-bf]+j];
						}
					}
				}

				// wait until gather completed
				//MPI_Wait(&mpi_request, &mpi_status);
				//TODO: substitute Gather with an All-to-All
				//MPI_Ibcast (&lastK[0][0], 2*n*bf, MPI_DOUBLE, map[l-1], comm, &mpi_request);
				MPI_Bcast (&lastK[0][0], 2*n*bf, MPI_DOUBLE, map[l-bf], comm);
			}
		}
	}

	// last level (l=0)
	for (i=0; i<myxxrows; i++)
	{
		for(rhs=0;rhs<m;rhs++)
		{
			xx[global[i]][rhs]=xx[global[i]][rhs]+Xlocal[0][i]*bb[0][rhs];
		}
	}

	//MPI_Wait(&mpi_request, &mpi_status);

    result.core_end_time = time(NULL);
	result.exit_code = 0;
/*
    for (i=0;i<cprocs;i++)
    {
    	MPI_Barrier(comm);
    	if(rank==i)
    	{
    	printf("%d:\n",rank);
    	printf("xx\n");
		PrintMatrix2D(xx, n, m);
		printf("\n");
    	}
    }
*/
	// collect solution
	// MPI_IN_PLACE required for MPICH based versions
	if (rank==0)
	{
		MPI_Gather (MPI_IN_PLACE, 1, xx_rows_interleaved_resized, &xx[0][0], 1, xx_rows_interleaved_resized, 0, comm);
	}
	else
	{
		MPI_Gather (&xx[rank*bf][0], 1, xx_rows_interleaved_resized, &xx[0][0], 1, xx_rows_interleaved_resized, 0, comm);
	}

	MPI_Datatype Tlocal_half;
	MPI_Type_vector (n*myKcols, 1, 1, MPI_DOUBLE, & Tlocal_half );
	MPI_Type_commit (& Tlocal_half);

	MPI_Datatype Thalf_interleaved;
	MPI_Type_vector (n*myKcols/bf, bf, bf*cprocs, MPI_DOUBLE, & Thalf_interleaved );
	MPI_Type_commit (& Thalf_interleaved);

	MPI_Datatype Thalf_interleaved_resized;
	MPI_Type_create_resized (Thalf_interleaved, 0, bf*sizeof(double), & Thalf_interleaved_resized);
	MPI_Type_commit (& Thalf_interleaved_resized);

	MPI_Gather (&Xlocal[0][0], 1, Tlocal_half, &A[0][0], 1, Thalf_interleaved_resized, 0, comm);

	MPI_Barrier(comm);
	if (rank==0 )
	{
		printf("\n\n Matrix X:\n");
		PrintMatrix2D(A, n, n);
		fflush(stdout);
	}
	// cleanup
	free(local);
	free(global);
	free(map);

	DeallocateMatrix2D(lastK,2,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Xlocal,n,CONTIGUOUS);
	DeallocateMatrix2D(Klocal,n,CONTIGUOUS);

	result.total_end_time = time(NULL);

	return result;
}
