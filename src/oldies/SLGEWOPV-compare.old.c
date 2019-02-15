#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../helpers/selfie.h"
#include "../helpers/matrix.h"
#include "../helpers/vector.h"
#include "../SLGEWOPV.h"
#include "GaussJordanElimination/GJE-par.h"
#include "Scalapack/ScalapackPDGESV.h"
#include <mpi.h>


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* get number of processes */

	//metrics program;

    int i,j,k,l,rep;

    double** A2;
    double*  b;
    double*  c;
    double*  x;

    //double** X;
    double** K;
    double*  H;
    double*  F;
    double*  s;

    double*  A1;
    int*     ipiv;

    /*
    double GJErunt[100];
    double LUt[100];
    double BSt[100];
    */
    double GJEt[100];
    double IMet[100];
    double IMelut[100];
    double IMesot[100];
    double LPKt[100];
    double GJEtotrunt=0.0;
    double IMetotrunt=0.0;
    double IMelutotrunt=0.0;
    double IMesototrunt=0.0;
    double LPKtotrunt=0.0;

    int n=atoi(argv[1]);

    int rows=n;
    int cols=n;
    int ft=atoi(argv[2]);
    int repetitions=atoi(argv[3]);
    int verbose=atoi(argv[4]);

    //time_t start1, stop1, start2, stop2, start3, stop3, start4, stop4, start5, stop5;
    clock_t start1, stop1, start2, stop2, start3, stop3, start4, stop4, start5, stop5, start6, stop6;

	if (rank==0 && verbose>0)
	{
		printf("\nMatrix size: %dx%d",n,n);
		printf("\nCheckpoint : ");
		if(ft==0)
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

    	// ScaLapack

		A1=AllocateMatrix1D(rows, cols);
		b=AllocateVector(rows);
		//ipiv = malloc(n * sizeof(int));

		if (rank==0)
		{
			FillMatrix1D(A1, rows, cols);
			FillVector(b,rows,1);

			if (verbose>2)
			{
				printf("\n\n Matrix A:\n");
				PrintMatrix1D(A1, rows, cols);
				printf("\n Vector b:\n");
				PrintVector(b, rows);
			}
		}

		start4=clock();
		//getmetrics(&program);
		//start4=program.wall_clock;

		ScalapackPDGESV_calc(A1, b, n, rank, nprocs);

		//getmetrics(&program);
		//stop4=program.wall_clock;
		stop4=clock();

		if (rank==0)
		{


			if (verbose>1)
			{
				printf("\nThe SPK solution is:\n");
				PrintVector(b, rows);
			}
		}
		DeallocateMatrix1D(A1);
		DeallocateVector(b);
		//free(ipiv);

		//////////////////////////////////////////////////////////////////////////////////
		// Gaussian Elimination

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

		start1 =clock();
		//getmetrics(&program);
		//start1=program.wall_clock;

		pGaussianElimination(A2, b, n,rank,nprocs);

	    if (rank==0)
		{
			BackSubstitution(A2, b, x, n);

			//getmetrics(&program);
			//stop2=program.wall_clock;
			stop2 = clock();

			if (verbose>1)
			{
				printf("\nThe GJE solution is:\n");
				PrintVector(x, rows);
			}
		}

	    DeallocateMatrix2D(A2,rows,CONTIGUOUS);
	    DeallocateVector(b);
	    DeallocateVector(x);

		//////////////////////////////////////////////////////////////////////////////////
		// Inhibition Method (base = broadcast interleaved)

	    A2=AllocateMatrix2D(rows,cols,CONTIGUOUS);
	    b=AllocateVector(rows);

		FillMatrix2D(A2, rows, cols);
		FillVector(b,rows,1);

	    //X=AllocateMatrix2D(n,n,CONTIGUOUS);
	    K=AllocateMatrix2D(n,n,CONTIGUOUS);
	    H=AllocateVector(n);
	    //F=AllocateVector(n);
	    s=AllocateVector(n);

		if (rank==0 && verbose>2)
		{
			printf("\n\n Matrix A:\n");
			PrintMatrix2D(A2, rows, cols);
			printf("\n Vector b:\n");
			PrintVector(b, rows);
		}

		start3 = clock();
		//getmetrics(&program);
		//start3=program.wall_clock;

		SLGEWOPV_calc(A2, b, s, n, K, H, rank, nprocs);

		//getmetrics(&program);
		//stop3=program.wall_clock;
		stop3 = clock();

		if (rank==0 && verbose>1)
		{
			printf("\nThe IMe solution is:\n");
			PrintVector(s, rows);
		}

	    DeallocateMatrix2D(A2,n,CONTIGUOUS);
	    //DeallocateMatrix2D(X,n,CONTIGUOUS);
	    DeallocateMatrix2D(K,n,CONTIGUOUS);
	    DeallocateVector(H);
	    //DeallocateVector(F);
	    DeallocateVector(s);
	    DeallocateVector(b);

	    //////////////////////////////////////////////////////////////////////////////////
		// Inhibition Method (loops unswitched)

		A2=AllocateMatrix2D(rows,cols,CONTIGUOUS);
		b=AllocateVector(rows);

		FillMatrix2D(A2, rows, cols);
		FillVector(b,rows,1);

		//X=AllocateMatrix2D(n,n,CONTIGUOUS);
		K=AllocateMatrix2D(n,n,CONTIGUOUS);
		H=AllocateVector(n);
		//F=AllocateVector(n);
		s=AllocateVector(n);

		if (rank==0 && verbose>2)
		{
			printf("\n\n Matrix A:\n");
			PrintMatrix2D(A2, rows, cols);
			printf("\n Vector b:\n");
			PrintVector(b, rows);
		}

		start5 = clock();
		//getmetrics(&program);
		//start5=program.wall_clock;

		SLGEWOPV_calc_unswitch(A2, b, s, n, K, H, rank, nprocs);

		//getmetrics(&program);
		//stop5=program.wall_clock;
		stop5 = clock();

		if (rank==0 && verbose>1)
		{
			printf("\nThe IMe-lu solution is:\n");
			PrintVector(s, rows);
		}

		DeallocateMatrix2D(A2,n,CONTIGUOUS);
		//DeallocateMatrix2D(X,n,CONTIGUOUS);
		DeallocateMatrix2D(K,n,CONTIGUOUS);
		DeallocateVector(H);
		//DeallocateVector(F);
		DeallocateVector(s);
		DeallocateVector(b);

		//////////////////////////////////////////////////////////////////////////////////
		// Inhibition Method (send optimized)

		A2=AllocateMatrix2D(rows,cols,CONTIGUOUS);
		b=AllocateVector(rows);

		FillMatrix2D(A2, rows, cols);
		FillVector(b,rows,1);

		//X=AllocateMatrix2D(n,n,CONTIGUOUS);
		K=AllocateMatrix2D(n,n,CONTIGUOUS);
		H=AllocateVector(n);
		//F=AllocateVector(n);
		s=AllocateVector(n);

		if (rank==0 && verbose>2)
		{
			printf("\n\n Matrix A:\n");
			PrintMatrix2D(A2, rows, cols);
			printf("\n Vector b:\n");
			PrintVector(b, rows);
		}

		start6 = clock();
		//getmetrics(&program);
		//start5=program.wall_clock;

		//SLGEWOPV_calc_unswitch(A2, b, s, n, K, H, rank, nprocs);
		SLGEWOPV_calc_sendopt(A2, b, s, n, K, H, rank, nprocs);

		//getmetrics(&program);
		//stop5=program.wall_clock;
		stop6 = clock();

		if (rank==0 && verbose>1)
		{
			printf("\nThe IMe-lu solution is:\n");
			PrintVector(s, rows);
		}

		DeallocateMatrix2D(A2,n,CONTIGUOUS);
		//DeallocateMatrix2D(X,n,CONTIGUOUS);
		DeallocateMatrix2D(K,n,CONTIGUOUS);
		DeallocateVector(H);
		//DeallocateVector(F);
		DeallocateVector(s);
		DeallocateVector(b);

		//////////////////////////////////////////////////////////////////////////////////

		if (rank==0)
		{
			/*
			LUt[rep]=(double)(end1 - begin1) / CLOCKS_PER_SEC;
			BSt[rep]=(double)(end2 - begin2) / CLOCKS_PER_SEC;
			IMet[rep]=(double)(end3 - begin3) / CLOCKS_PER_SEC;
			*/
			/*
			LUt[rep]=(double)(stop1 - start1);
			BSt[rep]=(double)(stop2 - start2);
			*/
			GJEt[rep]=(double)(stop2 - start1);
			IMet[rep]=(double)(stop3 - start3);
			LPKt[rep]=(double)(stop4 - start4);
			IMelut[rep]=(double)(stop5 - start5);
			IMesot[rep]=(double)(stop6 - start6);

			/*
			GJErunt[rep]=LUt[rep]+BSt[rep];
			GJEtotrunt += GJErunt[rep];
			*/
			GJEtotrunt += GJEt[rep];
			IMetotrunt += IMet[rep];
			LPKtotrunt += LPKt[rep];
			IMelutotrunt += IMelut[rep];
			IMesototrunt += IMesot[rep];

			/*
			printf("\n\nLU  decomposition time: %f", LUt[rep]);
			printf("\nBack substitution time: %f", BSt[rep]);
			printf("\nGaussian elimin.  time: %f\n", GJErunt[rep]);
			*/
			if(verbose>0)
			{
				printf("\nSPK    call    run time: %f clk", LPKt[rep]);
				printf("\nGJE    call    run time: %f clk", GJEt[rep]);
				printf("\nIMe    call    run time: %f clk", IMet[rep]);
				printf("\nIMe-lu call    run time: %f clk", IMelut[rep]);
				printf("\nIMe-so call    run time: %f clk\n", IMesot[rep]);
			}
		}
    }
	if (rank==0)
	{
		printf("\n Summary:");
		printf("\nSPK    Total   run time: %f clk\t\t%f s", LPKtotrunt, LPKtotrunt / CLOCKS_PER_SEC);
		printf("\nGJE    Total   run time: %f clk\t\t%f s", GJEtotrunt, GJEtotrunt / CLOCKS_PER_SEC);
		printf("\nIMe    Total   run time: %f clk\t\t%f s", IMetotrunt, IMetotrunt / CLOCKS_PER_SEC);
		printf("\nIMe-lu Total   run time: %f clk\t\t%f s", IMelutotrunt, IMelutotrunt / CLOCKS_PER_SEC);
		printf("\nIMe-so Total   run time: %f clk\t\t%f s\n", IMesototrunt, IMesototrunt / CLOCKS_PER_SEC);

		printf("\nSPK    Average run time: %f clk\t\t%f s", LPKtotrunt/repetitions, LPKtotrunt/repetitions / CLOCKS_PER_SEC);
		printf("\nGJE    Average run time: %f clk\t\t%f s", GJEtotrunt/repetitions, GJEtotrunt/repetitions / CLOCKS_PER_SEC);
		printf("\nIMe    Average run time: %f clk\t\t%f s", IMetotrunt/repetitions, IMetotrunt/repetitions / CLOCKS_PER_SEC);
		printf("\nIMe-lu Average run time: %f clk\t\t%f s", IMelutotrunt/repetitions, IMelutotrunt/repetitions / CLOCKS_PER_SEC);
		printf("\nIMe-so Average run time: %f clk\t\t%f s\n\n", IMesototrunt/repetitions, IMesototrunt/repetitions / CLOCKS_PER_SEC);
	}

	MPI_Finalize();
    return(0);
}
