#include <mpi.h>
#include "../helpers/matrix.h"
#include "../helpers/vector.h"
#include "../DGEZR.h"
#include "pDGEIT_Wx_ft1.h"

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

void pviDGESV_WO_ft1(int n, double** A, int m, double** bb, double** xx, int rank, int cprocs, int sprocs)
{
	int i,j,l;						// indexes
	int XKcols=2*n;					// num of cols X + K
    int myTcols;					// num of cols per process
    	myTcols=XKcols/cprocs;
    int Scols;
    	Scols=myTcols*sprocs;
    int Tcols=XKcols+Scols;			// num of cols X + K + S
    int myAchunks;					// num of A rows/cols per process
    	myAchunks=n/cprocs;
    int myxxrows=myAchunks;
    int myend;						// loop boundaries on local cols =myTcols/2;
    int mystart;
    int rhs;

    int avoidif;					// for boolean --> int conversion

    int fault_detected=0;

    int myAcols=n/cprocs;

    int nprocs=cprocs+sprocs;

    int ime_rank=rank;
    int* mpi_rank;
    mpi_rank=malloc(nprocs*sizeof(int));
    for (i=0;i<nprocs;i++)
    {
    	mpi_rank[i]=i;
    }

    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Tlocal;
    		 Tlocal=AllocateMatrix2D(n, myTcols, CONTIGUOUS);

	double** TlastK;
			 TlastK=AllocateMatrix2D(2,n, CONTIGUOUS);	// last col [0] and row [1] of T (K part)
	double*  TlastKc=&TlastK[0][0];						// alias for last col
	double*  TlastKr=&TlastK[1][0];						// alias for last row

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
					map[i]= cprocs + ((i-XKcols) % sprocs);		// n+1th has the first cols of T (S)
					local[i]= (i-XKcols) % sprocs;				// position of the column i(global) in the local matrix
				}
				for (i=0; i<XKcols; i++)
				{
					map[i]= i % cprocs;			// who has the other cols i (from ime_rank 1 onwards)
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
				if (ime_rank>=cprocs)
				{
					for(i=0; i<myTcols; i++)
					{
						global[i]= XKcols + i + (ime_rank-cprocs) * myTcols; 	// n+1th has the checksum cols (in the last positions of the column i(local) in the global matrix)
					}
				}
				else
				{
					for(i=0; i<myTcols; i++)
					{
						global[i]= i * cprocs + ime_rank; // position of the column i(local) in the global matrix
					}
				}
			}
			else									// without checksum cols
			{
				for(i=0; i<myTcols; i++)
				{
					global[i]= i * cprocs + ime_rank; 	// position of the column i(local) in the global matrix
				}
			}

    /*
     * MPI derived types
     */

	MPI_Datatype TlastKr_chunks;
	MPI_Type_vector (n/cprocs, 1, cprocs, MPI_DOUBLE, & TlastKr_chunks );
	MPI_Type_commit (& TlastKr_chunks);

	MPI_Datatype TlastKr_chunks_resized;
	MPI_Type_create_resized (TlastKr_chunks, 0, 1*sizeof(double), & TlastKr_chunks_resized);
	MPI_Type_commit (& TlastKr_chunks_resized);

	// rows of xx to be extracted
	MPI_Datatype xx_rows_interleaved;
	MPI_Type_vector (myxxrows, m, m*cprocs, MPI_DOUBLE, & xx_rows_interleaved );
	MPI_Type_commit (& xx_rows_interleaved);

	// rows of xx to be extracted, properly resized for gathering
	MPI_Datatype xx_rows_interleaved_resized;
	MPI_Type_create_resized (xx_rows_interleaved, 0, m*sizeof(double), & xx_rows_interleaved_resized);
	MPI_Type_commit (& xx_rows_interleaved_resized);

	int i_am_calc;
		if (ime_rank>=cprocs)
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
		MPI_Comm_split(MPI_COMM_WORLD, i_am_calc, ime_rank, &comm_calc);
		MPI_Comm_rank(comm_calc, &rank_calc);
	}
	else
	{
		comm_calc=MPI_COMM_WORLD;
		rank_calc=ime_rank;
	}

    /*
	 *  init inhibition table
	 */
	DGEZR(xx, n, m);													// init (zero) solution vectors
	pDGEIT_W_ft1(A, Tlocal, TlastK, n, ime_rank, cprocs, sprocs, map, global, local);	// init inhibition table

	// send all r.h.s to all procs
    MPI_Bcast (&bb[0][0], n*m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/*
	 *  calc inhibition sequence
	 */

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		if (l<4 && sprocs>0)	// if FT enabled, inject fault
		{
			// inject fault in ime_rank 1
			if (fault_detected)
			{
				// signal error persist
				//printf("\nFT: rank %d(%d) still faulty at level %d", ime_rank, mpi_rank[ime_rank], l);
			}
			else // first detection
			{
				printf("\nFT: mpi rank %d(ime %d) faulty at level %d", mpi_rank[ime_rank], ime_rank, l);
				fault_detected=1;
				printf("\nfirst fault detected %d", fault_detected);
				// swap ime_rank 1 (faulty) with last ime_rank (checksum)
				mpi_rank[1]=cprocs;
				mpi_rank[cprocs]=1;
				if (ime_rank==1)
				{
					i_am_calc=0;
					ime_rank=cprocs;
				}
				else
				{
					if (ime_rank==cprocs)
					{
						i_am_calc=1;
						ime_rank=1;
						for(i=0; i<myTcols; i++)
						{
							global[i]= i * cprocs + ime_rank; // position of the column i(local) in the global matrix
						}
					}
				}
				MPI_Comm_split(MPI_COMM_WORLD, i_am_calc, mpi_rank[ime_rank], &comm_calc);
				MPI_Comm_rank(comm_calc, &rank_calc);
			}
		}

		printf("\nfault detected %d", fault_detected);
		if ((fault_detected) && (ime_rank==1))	// if a fault is detected and I'm the faulty node
		{
			printf("\nI'm %d(%d) at level %d and I'm faulty", ime_rank, mpi_rank[ime_rank], l);
			// do nothing
		}
		else							// a fault is not detected or I'm not the faulty node
		{	// do as usual
			printf("\nI'm %d(%d) at level %d and I'm doing as usual", ime_rank, mpi_rank[ime_rank], l);
			// ALL procs
			// update helpers
			for (i=0; i<=l-1; i++)
			{
				h[i]   = 1/(1-TlastKc[i]*TlastKr[i]);
				hh[i]  = TlastKc[i]*h[i];
				for (rhs=0;rhs<m;rhs++)
				{
					bb[i][rhs] = bb[i][rhs]-TlastKr[i]*bb[l][rhs];
				}
			}

			if (ime_rank<cprocs)
			{
				// ALL procs
				// update solutions
				// l .. n-1

				/*
				mystart=local[l];
				if (ime_rank<map[l])
				{
					mystart++;
				}
				*/
				avoidif=ime_rank<map[l];
				mystart = local[l] + avoidif;

				for (i=mystart; i<=local[n-1]; i++)
				{
					for (rhs=0;rhs<m;rhs++)
					{
						xx[global[i]][rhs]=xx[global[i]][rhs]+Tlocal[l][i]*bb[l][rhs];
					}
				}

				// update T
				// to avoid IFs: each process loops on its own set of cols, with indirect addressing

				// 0 .. l-1
				// ALL procs
				// processes with diagonal elements not null
				mystart=0;
				/*
				myend=local[l-1];
				if (ime_rank>map[l-1])
				{
					myend--;
				}
				*/
				avoidif = ime_rank>map[l-1];
				myend = local[l-1] - avoidif;

				for (i=mystart; i<=myend; i++)
				{
					Tlocal[global[i]][i]=Tlocal[global[i]][i]*h[global[i]];
				}

				// l .. n+l-1
				// ALL procs
				/*
				mystart=local[l];
				if (ime_rank<map[l])
				{
					mystart++;
				}
				*/
				avoidif=ime_rank<map[l];
				mystart=local[l]+avoidif;
				/*
				myend=local[n+l-1];
				if (ime_rank>map[n+l-1])
				{
					myend--;
				}
				*/
				avoidif=ime_rank>map[n+l-1];
				myend=local[n+l-1]-avoidif;

				for (i=0; i<=l-1; i++)
				{
					for (j=mystart; j<=myend; j++)
					{
						Tlocal[i][j]=Tlocal[i][j]*h[i] - Tlocal[l][j]*hh[i];
					}
				}

				// collect chunks of last row of K to "future" last node
				MPI_Gather (&Tlocal[l-1][local[n]], myTcols/2, MPI_DOUBLE, &TlastKr[0], 1, TlastKr_chunks_resized, mpi_rank[map[l-1]], comm_calc);

			}
			else // node containing S
			{
				for (i=0; i<=l-1; i++)
				{
					for (j=0; j<myTcols; j++)
					{
						Tlocal[i][j]=Tlocal[i][j]*h[i]-Tlocal[l][j]*hh[i];
					}
				}
			}

			//future last node broadcasts last row and col of K
			if (ime_rank==map[n+l-1]) // n
			{
				// copy data into local buffer before broadcast
				for (i=0; i<n; i++)
				{
					TlastKc[i]=Tlocal[i][local[n+l-1]];
				}
			}

			//TODO: substitute Gather with an All-to-All
			//MPI_Bcast (&TlastK[0][0], Tcols, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
			//MPI_Bcast (&TlastK[0][0], XKcols, MPI_DOUBLE, mpi_rank[map[n+l-1]], comm_calc); // n

		}// done as usual

	}// end loop over levels > 0

	printf("\nmain loop ended for %d(%d)", ime_rank, mpi_rank[ime_rank]);
	// last level (l=0)
	if (i_am_calc)
	{

		for (i=0; i<myTcols/2; i++)
		{
			for(rhs=0;rhs<m;rhs++)
			{
				xx[global[i]][rhs]=xx[global[i]][rhs]+Tlocal[0][i]*bb[0][rhs];
			}
		}

		// collect solution
		// MPI_IN_PLACE required for MPICH based versions
		if (ime_rank==0)
		{
			MPI_Gather (MPI_IN_PLACE, 1, xx_rows_interleaved_resized, &xx[0][0], 1, xx_rows_interleaved_resized, mpi_rank[0], comm_calc);
		}
		else
		{
			MPI_Gather (&xx[ime_rank][0], 1, xx_rows_interleaved_resized, &xx[0][0], 1, xx_rows_interleaved_resized, mpi_rank[0], comm_calc);
		}
	}



	/*
	MPI_Barrier(MPI_COMM_WORLD);
	sleep(2*ime_rank);
	printf("\n\n Matrix Tloc (%d):\n",ime_rank);
	PrintMatrix2D(Tlocal, n, myTcols);
	MPI_Barrier(MPI_COMM_WORLD);
	*/

	// cleanup
	free(local);
	free(global);
	free(map);
	free(mpi_rank);

	DeallocateMatrix2D(TlastK,2,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Tlocal,n,CONTIGUOUS);

	MPI_Type_free(&TlastKr_chunks);
	MPI_Type_free(&TlastKr_chunks_resized);
	MPI_Type_free(&xx_rows_interleaved);
	MPI_Type_free(&xx_rows_interleaved_resized);
	if (sprocs>0)
	{
		MPI_Comm_free(&comm_calc);
	}
}
