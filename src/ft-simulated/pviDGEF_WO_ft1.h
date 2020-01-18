#include <unistd.h>
#include <mpi.h>
#include "../helpers/matrix.h"
#include "../helpers/vector.h"
#include "pDGEIT_WX_ft1.h"

/*
 *	factorization (F) of a general (GE) matrix A of doubles (D)
 *	of order n
 *	with:
 *	wide overwrite (WO) memory model
 *	parallelized in interleaved columns (pvi) over cprocs calculating processors
 *	parallelized initialization
 *	optimized loops
 *	ifs removed
 *
 */

void pviDGEF_WO_ft1_sim(int n, double** A, double** K, MPI_Comm comm, int sprocs, int failing_rank, int failing_level)
{
    int rank, nprocs, cprocs; //
    MPI_Comm_rank(comm, &rank);		//get current process id
    MPI_Comm_size(comm, &nprocs);	// get number of processes
    cprocs = nprocs - sprocs;

	int i,j,l;						// indexes
	int XKcols=2*n;					// num of cols X + K
    int myTcols;					// num of cols per process
    	myTcols=XKcols/cprocs;
    int myKcols;
    	myKcols=myTcols/2;
    int Scols;
    	Scols=myTcols*sprocs;
    int Tcols=XKcols+Scols;			// num of cols X + K + S

    int myend;						// loop boundaries on local cols =myTcols/2;
    int mystart;

    int avoidif;					// for boolean --> int conversion

    int fault_detected=0;


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

			for (i=0; i<XKcols; i++)
			{
				map[i]= i % cprocs;			// who has the col i
				local[i]=(int)floor(i/cprocs);	// position of the column i(global) in the local matrix
			}
			for (i=0; i<Scols; i++)			// not executed if sprocs = 0
			{
				map[XKcols+i]= cprocs + (i % sprocs);		// n+1th has the first cols of T (S)
				local[XKcols+i]= i % sprocs;				// position of the column i(global) in the local matrix
			}

	int*	global;
			global=malloc(myTcols*sizeof(int));

			for(i=0; i<myTcols; i++)		// executed also on ranks >= cprocs, but useless
			{
				global[i]= i * cprocs + rank; 	// position of the column i(local) in the global matrix
			}

    /*
     * MPI derived types
     */
	MPI_Datatype Tlocal_half;
	MPI_Type_vector (n, myKcols, myTcols, MPI_DOUBLE, & Tlocal_half );
	MPI_Type_commit (& Tlocal_half);

	MPI_Datatype Thalf_interleaved;
	MPI_Type_vector (n*myKcols, 1, cprocs, MPI_DOUBLE, & Thalf_interleaved );
	MPI_Type_commit (& Thalf_interleaved);

	MPI_Datatype Thalf_interleaved_resized;
	MPI_Type_create_resized (Thalf_interleaved, 0, 1*sizeof(double), & Thalf_interleaved_resized);
	MPI_Type_commit (& Thalf_interleaved_resized);
	/*
	 * option 1: derived data type, for interleaving while for gathering
	 *           non working for some values of n and np, why? extent of derived MPI data type?
	 */
	/*
	MPI_Datatype TlastKr_chunks;
	MPI_Type_vector (n/cprocs, 1, cprocs, MPI_DOUBLE, & TlastKr_chunks );
	MPI_Type_commit (& TlastKr_chunks);

	MPI_Datatype TlastKr_chunks_resized;
	MPI_Type_create_resized (TlastKr_chunks, 0, 1*sizeof(double), & TlastKr_chunks_resized);
	MPI_Type_commit (& TlastKr_chunks_resized);
	*/

	/*
	 * option 2: standard data type, explicit loop for interleaving after gathering
	 *           see below
	 */

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
		current_comm_world=comm;
		MPI_Comm_split(comm, MPI_UNDEFINED, MPI_UNDEFINED, &comm_calc); // checksumming procs don't belong to calc communicator
	}
	else
	{
		i_am_calc=1;
		i_am_fine=1;
		current_comm_world=comm;
		MPI_Comm_split(comm, i_am_calc, rank, &comm_calc); // calc procs belong to calc communicator
	}

    /*
	 *  init inhibition table
	 */
	pDGEIT_W_ft1(A, Tlocal, TlastK, n, rank, cprocs, sprocs, map, global, local);	// init inhibition table

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

			if (rank==failing_rank)
			{
				printf("\n## IMe pviDGEF: bye bye from %d",rank);

				i_am_fine=0;
				i_am_calc=0;
				// simulate recovery
				//MPI_Send(&xx[0][0], n*m, MPI_DOUBLE, spare_rank, 0, MPI_COMM_WORLD);
				MPI_Comm_split(comm, MPI_UNDEFINED, MPI_UNDEFINED, &comm_calc);

				//sleep(2);
				break;
			}

			if (rank==spare_rank)
			{
				printf("\n## IMe pviDGEF: hello from %d, ",rank);

				i_am_calc=1;
				// simulate recovery
				//MPI_Recv(&xx[0][0], n*m, MPI_DOUBLE, failing_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				MPI_Comm_split(comm, i_am_fine, failing_rank, &comm_calc);
				rank=failing_rank;
				printf("now %d",rank);
				printf("\n## IMe pviDGEF: recovering..");

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
				for (i=0;i<n;i++)															// recovery = - recovery
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
				/*
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
				*/
				printf("\n## IMe pviDGEF: ..recovered!\n");

			} // spare rank

			if ((rank!=spare_rank) && (rank!=failing_rank))
			{
				MPI_Comm_split(comm, i_am_fine, rank, &comm_calc);
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
		}

		if (i_am_calc)
		{
			// update T
			// to avoid IFs: each process loops on its own set of cols, with indirect addressing

			// 0 .. l-1
			// ALL procs
			// processes with diagonal elements not null
			mystart=0;
			avoidif = rank>map[l-1];
			myend = local[l-1] - avoidif;

			for (i=mystart; i<=myend; i++)
			{
				Tlocal[global[i]][i]=Tlocal[global[i]][i]*h[global[i]];
			}

			// l .. n+l-1
			// ALL procs
			avoidif=rank<map[l];
			mystart=local[l]+avoidif;
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

			// option 1: non working
			//MPI_Gather (&Tlocal[l-1][local[n]], myTcols/2, MPI_DOUBLE, &TlastKr[0], 1, TlastKr_chunks_resized, map[l-1], comm_calc);
			// option 2: use last column buffer for temporary copy of non-interleaved data
			MPI_Gather (&Tlocal[l-1][myKcols], myKcols, MPI_DOUBLE, &TlastKc[0], myKcols, MPI_DOUBLE, map[l-1], comm_calc);
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
		if (rank==map[l-1]) // n
		{
			// copy data into local buffer before broadcast

			// option 1
			/*
			for (i=0; i<n; i++)
			{
				TlastKc[i]=Tlocal[i][local[n+l-1]];
			}
			*/

			// option 2
			myend=local[n+l-1];
			for (i=0; i<myKcols; i++)
			{
				int ii=i*cprocs;
				for (j=0; j<cprocs; j++)
				{
					int jj=j*myKcols+i;
					TlastKr[ii+j]=TlastKc[jj];		// interleave columns of the last row
					TlastKc[jj]=Tlocal[jj][myend];	// copy last column
				}
			}
		}

		//TODO: substitute Gather with an All-to-All
		MPI_Bcast (&TlastK[0][0], XKcols, MPI_DOUBLE, map[l-1], current_comm_world); // n


	}// end loop over levels > 0


	if (i_am_calc)
	{
		MPI_Gather (&Tlocal[0][0], 1, Tlocal_half, &A[0][0], 1, Thalf_interleaved_resized, 0, comm_calc);
		MPI_Gather (&Tlocal[0][myKcols], 1, Tlocal_half, &K[0][0], 1, Thalf_interleaved_resized, 0, comm_calc);
	}

	// cleanup
	free(local);
	free(global);
	free(map);

	DeallocateMatrix2D(TlastK,2,CONTIGUOUS);
	DeallocateVector(h);
	DeallocateVector(hh);
	DeallocateMatrix2D(Tlocal,n,CONTIGUOUS);

	// option 1
	/*
	MPI_Type_free(&TlastKr_chunks);
	MPI_Type_free(&TlastKr_chunks_resized);
	*/
	if (sprocs>0)
	{
		if (comm_calc != MPI_COMM_NULL)
		{
			MPI_Comm_free(&comm_calc);
		}
	}
}

