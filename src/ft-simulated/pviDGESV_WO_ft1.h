#include <unistd.h>
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

void pviDGESV_WO_ft1_sim(int n, double** A, int m, double** bb, double** xx, int rank, int cprocs, int sprocs, int failing_rank, int failing_level)
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


    /*
     * local storage for a part of the input matrix (continuous columns, not interleaved)
     */
    double** Tlocal;
    		 Tlocal=AllocateMatrix2D(n, myTcols, CONTIGUOUS);

	double** TlastK;
			 TlastK=AllocateMatrix2D(2,n, CONTIGUOUS);	// last col [0] and row [1] of T (K part)
	double*  TlastKc=&TlastK[0][0];						// alias for last col
	double*  TlastKr=&TlastK[1][0];						// alias for last row

    double* h;											// helper vectors
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
			/*
			 * old version

			if (sprocs>0)						// with checksum cols
			{
				for (i=XKcols; i<Tcols; i++)
				{
					map[i]= cprocs + ((i-XKcols) % sprocs);		// n+1th has the first cols of T (S)
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
			*/
			for (i=0; i<XKcols; i++)
			{
				map[i]= i % cprocs;			// who has the col i
				local[i]=floor(i/cprocs);	// position of the column i(global) in the local matrix
			}
			for (i=0; i<Scols; i++)			// not executed if sprocs = 0
			{
				map[XKcols+i]= cprocs + (i % sprocs);		// n+1th has the first cols of T (S)
				local[XKcols+i]= i % sprocs;				// position of the column i(global) in the local matrix
			}

	int*	global;
			global=malloc(myTcols*sizeof(int));
			/*
			 * old version

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
			*/
			for(i=0; i<myTcols; i++)		// executed also on ranks >= cprocs, but useless
			{
				global[i]= i * cprocs + rank; 	// position of the column i(local) in the global matrix
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

	/*
	 * distinction between calc/checksumming, healthy/faulty
	 */
	MPI_Comm current_comm_world, comm_calc;

	int i_am_calc; // participating in ime calc = 1, checksumming = 0
	int i_am_fine; // healthy = 1, faulty = 0
	int spare_rank=cprocs;

	if (rank>=cprocs)
	{
		i_am_calc=0;
		i_am_fine=1;
		current_comm_world=MPI_COMM_WORLD;
		MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, MPI_UNDEFINED, &comm_calc); // checksumming procs don't belong to calc communicator
	}
	else
	{
		i_am_calc=1;
		i_am_fine=1;
		current_comm_world=MPI_COMM_WORLD;
		MPI_Comm_split(MPI_COMM_WORLD, i_am_calc, rank, &comm_calc); // calc procs belong to calc communicator
	}

    /*
	 *  init inhibition table
	 */
	DGEZR(xx, n, m);																			// init (zero) solution vectors
	pDGEIT_W_ft1(A, Tlocal, TlastK, n, rank, cprocs, sprocs, map, global, local, failing_rank);	// init inhibition table
    MPI_Bcast(&bb[0][0], n*m, MPI_DOUBLE, 0, MPI_COMM_WORLD);									// send all r.h.s to all procs

	/*
	 *  calc inhibition sequence
	 */

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		if ((l<=failing_level) && (sprocs>0) && (fault_detected==0)) // time to have a fault!
		{
			double** Slocal;
			Slocal=AllocateMatrix2D(n,myTcols, CONTIGUOUS);

			fault_detected=1;
			/*
			 * old version

			if (rank==failing_rank)
			{
				i_am_fine=0;
			}
			if (i_am_fine)
			{
				if (rank<cprocs)
				{
					MPI_Comm_split(MPI_COMM_WORLD, i_am_fine, rank, &comm_calc);
				}
				else // (rank == cprocs)
				{
					i_am_calc=1;
					MPI_Comm_split(MPI_COMM_WORLD, i_am_fine, failing_rank, &comm_calc);
				}
			}
			else // I'm faulty
			{
				MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, MPI_UNDEFINED, &comm_calc);
			}

			if (rank==failing_rank)	// if FT enabled, inject fault
			{
				MPI_Send(&xx[0][0], n*m, MPI_DOUBLE, cprocs, 0, MPI_COMM_WORLD);

				printf("\n bye bye from %d",rank);
				sleep(2);
				break;
			}
			else
			{
				// swap rank 1 (faulty) with last rank (checksum)
				if (rank==cprocs) // new 1
				{
					MPI_Recv(&xx[0][0], n*m, MPI_DOUBLE, failing_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

					printf("\n hello from %d, ",rank);
					rank=failing_rank;
					printf("now %d",rank);
					for(i=0; i<myTcols; i++)
					{
						global[i]= i * cprocs + rank; 	// position of the column i(local) in the global matrix
					}
				}
			}
			current_comm_world=comm_calc;
			MPI_Comm_rank(current_comm_world, &rank);	//get current process id
			*/
			if (rank==failing_rank)
			{
				printf("\n bye bye from %d",rank);

				i_am_fine=0;
				i_am_calc=0;
				// simulate recovery
				//MPI_Send(&xx[0][0], n*m, MPI_DOUBLE, spare_rank, 0, MPI_COMM_WORLD);
				MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, MPI_UNDEFINED, &comm_calc);

				sleep(2);
				break;
			}
			if (rank==spare_rank)
			{
				printf("\n hello from %d, ",rank);

				i_am_calc=1;
				// simulate recovery
				//MPI_Recv(&xx[0][0], n*m, MPI_DOUBLE, failing_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				MPI_Comm_split(MPI_COMM_WORLD, i_am_fine, failing_rank, &comm_calc);
				rank=failing_rank;
				printf("now %d\n",rank);
				printf("\n recovering..");

				// recovery from other procs
				// recovery = checksum - node1 - node2 - .. = - (- checksum + node1 + node2 +..)
				for (i=0;i<n;i++)																			// checksum = - checksum
				{
					for (j=0;j<myTcols;j++)
					{
						Tlocal[i][j]=-1*Tlocal[i][j];
					}
				}
				MPI_Reduce(&Tlocal[0][0], &Slocal[0][0], n*myTcols, MPI_DOUBLE, MPI_SUM, failing_rank, comm_calc);	// recovery = - checksum + node1 + node2 +..
				for (i=0;i<n;i++)for (i=0;i<n;i++)															// recovery = - recovery
				{
					for (j=0;j<myTcols;j++)
					{
						Tlocal[i][j]=-1*Slocal[i][j];
					}
				}

				// recovery from myself
				for(i=0; i<myTcols; i++)			// update index with future new rank
				{
					global[i]= i * cprocs + rank; 	// position of the column i(local) in the global matrix
				}
				for (j=n-1; j>failing_level; j--)
				{
					avoidif=rank<map[j];
					mystart = local[j] + avoidif;
					for (i=mystart; i<=local[n-1]; i++)
					{
						for (rhs=0;rhs<m;rhs++)
						{
							xx[global[i]][rhs]=xx[global[i]][rhs]+Tlocal[j][i]*bb[j][rhs];
						}
					}
				}
				printf("\n ..recovered!\n");

			} // spare rank

			if ((rank!=spare_rank) && (rank!=failing_rank))
			{
				MPI_Comm_split(MPI_COMM_WORLD, i_am_fine, rank, &comm_calc);
				MPI_Reduce(&Tlocal[0][0], &Slocal[0][0], n*myTcols, MPI_DOUBLE, MPI_SUM, failing_rank, comm_calc);	// recovery = - checksum + node1 + node2 +..
			}
			current_comm_world=comm_calc;
			MPI_Comm_rank(current_comm_world, &rank);

		} // time to have a fault!

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

		if (i_am_calc)
		{
			// ALL procs
			// update solutions
			// l .. n-1

			/*
			mystart=local[l];
			if (rank<map[l])
			{
				mystart++;
			}
			*/
			avoidif=rank<map[l];
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
			if (rank>map[l-1])
			{
				myend--;
			}
			*/
			avoidif = rank>map[l-1];
			myend = local[l-1] - avoidif;

			for (i=mystart; i<=myend; i++)
			{
				Tlocal[global[i]][i]=Tlocal[global[i]][i]*h[global[i]];
			}

			// l .. n+l-1
			// ALL procs
			/*
			mystart=local[l];
			if (rank<map[l])
			{
				mystart++;
			}
			*/
			avoidif=rank<map[l];
			mystart=local[l]+avoidif;
			/*
			myend=local[n+l-1];
			if (rank>map[n+l-1])
			{
				myend--;
			}
			*/
			avoidif=rank>map[n+l-1];
			myend=local[n+l-1]-avoidif;

			for (i=0; i<=l-1; i++)
			{
				for (j=mystart; j<=myend; j++)
				{
					Tlocal[i][j]=Tlocal[i][j]*h[i] - Tlocal[l][j]*hh[i];
				}
			}

			// collect chunks of last row of K to "future" last node
			MPI_Gather (&Tlocal[l-1][local[n]], myTcols/2, MPI_DOUBLE, &TlastKr[0], 1, TlastKr_chunks_resized, map[l-1], comm_calc);
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
		if (rank==map[n+l-1]) // n
		{
			// copy data into local buffer before broadcast
			for (i=0; i<n; i++)
			{
				TlastKc[i]=Tlocal[i][local[n+l-1]];
			}
		}

		//TODO: substitute Gather with an All-to-All
		//MPI_Bcast (&TlastK[0][0], Tcols, MPI_DOUBLE, map[l-1], MPI_COMM_WORLD);
		MPI_Bcast (&TlastK[0][0], XKcols, MPI_DOUBLE, map[l-1], current_comm_world); // n


	}// end loop over levels > 0

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

		//if (comm_calc != MPI_COMM_NULL)
		{
			if (rank==0)
			{
				MPI_Gather (MPI_IN_PLACE, 1, xx_rows_interleaved_resized, &xx[0][0], 1, xx_rows_interleaved_resized, 0, comm_calc);
			}
			else
			{
				MPI_Gather (&xx[rank][0], 1, xx_rows_interleaved_resized, &xx[0][0], 1, xx_rows_interleaved_resized, 0, comm_calc);
			}
		}
	}

	// cleanup
	free(local);
	free(global);
	free(map);

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
		if (comm_calc != MPI_COMM_NULL)
		{
			MPI_Comm_free(&comm_calc);
		}
	}
}

