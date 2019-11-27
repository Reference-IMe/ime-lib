#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../helpers/matrix.h"
#include "../helpers/vector.h"

#include "test_Scalapack_pDGESV.h"
#include "test_GaussianElimination_pGE.h"
#include "test_GaussianElimination_pGE_cp.h"
#include "test_IMe_pviDGESV.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, totprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);		/* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &totprocs);	/* get number of processes */

    int i,rep;

    double versionrun[10][100];
    const char* versionname[10];
    double versiontot[10];

    int n=atoi(argv[1]);

    int rows=n;
    int cols=n;
    int sprocs=atoi(argv[2]);		// number of processes to allocate for summing (0 = no fault tolerance)
    int cprocs=totprocs-sprocs;		// number of processes for real IMe calc
    int repetitions=atoi(argv[3]);
    int verbose=atoi(argv[4]);

    int nRHS=10;

	versionname[0]="SPK     1 ";
	versionname[1]="SPK     10";
	versionname[2]="GE      1 ";
	versionname[3]="GE      10";
	versionname[4]="GE-cp0  1 ";
	versionname[5]="GE-cp0  10";
	versionname[6]="IMe     1 ";
	versionname[7]="IMe     10";

	int versions = 8;

	for (i=0; i<versions; i++)
	{
		versiontot[i] = 0;
	}

	if (rank==0 && verbose>0)
	{
		printf("\nMatrix size: %dx%d",n,n);
		printf("\nCheckpoint : ");
		if(sprocs==0)
		{
			printf("no\n");
		}
		else
		{
			printf("yes\n");
		}
	}

    for (rep=0; rep<repetitions; rep++)
    {
    	if (rank==0 && verbose>0) {printf("\n Run #%d",rep+1);}

    	//////////////////////////////////////////////////////////////////////////////////

    	versionrun[0][rep]=test_Scalapack_pDGESV(versionname[0], verbose, rows, cols, 1, rank, cprocs);    					// Scalapack with 1 rhs
    	versionrun[1][rep]=test_Scalapack_pDGESV(versionname[1], verbose, rows, cols, nRHS, rank, cprocs);					// Scalapack with 10 rhs
    	versionrun[2][rep]=test_pGaussianElimination(versionname[2], verbose, rows, cols, 1, rank, cprocs);    				// GE with 1 rhs
    	versionrun[3][rep]=test_pGaussianElimination(versionname[3], verbose, rows, cols, nRHS, rank, cprocs);				// GE with 10 rhs
    	versionrun[4][rep]=test_pGaussianElimination_cp(versionname[4], verbose, rows, cols, 1, rank, cprocs, sprocs); 		// GE checkpointed with 1 rhs
    	versionrun[5][rep]=test_pGaussianElimination_cp(versionname[5], verbose, rows, cols, nRHS, rank, cprocs, sprocs);	// GE checkpointed with 10 rhs
    	versionrun[6][rep]=test_IMe_pviDGESV(versionname[6], verbose, rows, cols, 1, rank, cprocs);    						// IMe with 1 rhs
    	versionrun[7][rep]=test_IMe_pviDGESV(versionname[7], verbose, rows, cols, nRHS, rank, cprocs);						// IMe with 10 rhs

		//////////////////////////////////////////////////////////////////////////////////

    	if (rank==0)
		{
			for (i=0; i<versions; i++)
			{
				versiontot[i] += versionrun[i][rep];
				if (verbose>0)
				{
					printf("\n%s    call    run time: %f clk", versionname[i], versionrun[i][rep]);
				}
			}
		}
    }

	if (rank==0)
	{
		printf("\n\n Summary:");
		for (i=0; i<versions; i++)
		{
			printf("\n%s    Total   run time: %f clk\t\t%f s", versionname[i], versiontot[i], versiontot[i] / CLOCKS_PER_SEC);
		}
		printf("\n");
		for (i=0; i<versions; i++)
			{
				printf("\n%s    Average run time: %f clk\t\t%f s", versionname[i], versiontot[i]/repetitions, versiontot[i]/repetitions / CLOCKS_PER_SEC);
			}
		printf("\n");
	}

	MPI_Finalize();
    return(0);
}
