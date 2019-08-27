#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../helpers/matrix.h"
#include "../helpers/vector.h"

#include <mpi.h>

#include "../SLGEWOPV-FT.h"
#include "../SLGEWOPV-mFT.h"
#include "GaussianElimination/pGE.h"
#include "Scalapack/Scalapack_pDGESV.h"



int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, totprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &totprocs); /* get number of processes */

    int i,j,k,l,rep;

    double** A2;
    double** T;
    double*  b;
    double*  c;
    double*  x;

    double** K;
    double*  H;
    //double*  F;

    double*  A1;
    int*     ipiv;

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

    clock_t start, stop;

    versionname[0]="GJE-cp  ";
    versionname[1]="IMe-cs  ";

	int versions = 2;

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
		// Gaussian Elimination (with checkpointing)

		A2=AllocateMatrix2D(rows,cols,CONTIGUOUS);
		b=AllocateVector(rows);
		x=AllocateVector(rows);

		if (rank==0)
		{
			FillMatrix2D(A2, rows, cols);
			FillVector(b,rows,1);

			if (verbose>2)
			{
				printf("\n\n Matrix A:\n");
				PrintMatrix2D(A2, rows, cols);
				printf("\n Vector b:\n");
				PrintVector(b, rows);
			}
		}

		start=clock();

		pGaussianElimination_partialmatrix_cp(A2, b, n, rank, cprocs, sprocs);

		if (rank==0)
		{
			BackSubstitution(A2, b, x, n);
			stop=clock();
			versionrun[0][rep]=(double)(stop - start);

			if (verbose>1)
			{
				printf("\nThe %s solution is:\n",versionname[0]);
				PrintVector(x, rows);
			}
		}

		DeallocateMatrix2D(A2,rows,CONTIGUOUS);
		DeallocateVector(b);
		DeallocateVector(x);

		//////////////////////////////////////////////////////////////////////////////////
		// Inhibition Method (last with checksumming)

		x=AllocateVector(n);
		b=AllocateVector(rows);

		if (rank==0)
		{
			A2=AllocateMatrix2D(rows,cols,CONTIGUOUS);

			FillMatrix2D(A2, rows, cols);

			FillVector(b,rows,1);

			if (verbose>2)
			{
				printf("\n\n Matrix A:\n");
				PrintMatrix2D(A2, rows, cols);
				printf("\n Vector b:\n");
				PrintVector(b, rows);
			}
		}

		start = clock();

		//SLGEWOPV_calc_last_cs(A2, b, x, n, rank, cprocs, sprocs, T, Tlocal, TlastKc, TlastKr, h, hh);
		SLGEWOPV_calc_last_cs(A2, b, x, n, rank, cprocs, sprocs);

		stop = clock();
		versionrun[1][rep]=(double)(stop - start);

		if (rank==0 && verbose>1)
		{
			printf("\nThe %s solution is:\n",versionname[1]);
			PrintVector(x, rows);
		}

		DeallocateVector(x);
		DeallocateVector(b);
		if (rank==0)
		{
			DeallocateMatrix2D(A2,n,CONTIGUOUS);
		}

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
