#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../helpers/matrix.h"
#include "../helpers/vector.h"

#include "test_GaussianElimination_GE.h"
#include "test_Lapack_DGESV.h"
#include "test_IMe_DGESV.h"

int main(int argc, char **argv)
{
    int i,rep;

    int n=atoi(argv[1]);
    int rows=n;
    int cols=n;

    int ft=atoi(argv[2]);

    int repetitions=atoi(argv[3]);

    int verbose=atoi(argv[4]);

    int nRHS=10;

    double versionrun[10][100];
    double versiontot[10];
    const char* versionname[10];
	versionname[0]="LPK   1 ";
	versionname[1]="LPK   10";
	versionname[2]="GE    1 ";
	versionname[3]="GE    10";
	versionname[4]="IMe   1 ";
	versionname[5]="IMe   10";
	int versions = 6;

	for (i=0; i<versions; i++)
	{
		versiontot[i] = 0;
	}

	if (verbose>0)
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
    	if (verbose>0) {printf("\n\n Run #%d",rep+1);}

    	//////////////////////////////////////////////////////////////////////////////////

    	versionrun[0][rep]=test_Lapack_DGESV(versionname[0], verbose, rows, cols, 1);    			// Lapack with 1 rhs
    	versionrun[1][rep]=test_Lapack_DGESV(versionname[1], verbose, rows, cols, nRHS);			// Lapack with 10 rhs
    	versionrun[2][rep]=test_GaussianElimination(versionname[2], verbose, rows, cols, 1);		// Gaussian Elimination with 1 rhs
    	versionrun[3][rep]=test_GaussianElimination(versionname[3], verbose, rows, cols, nRHS);		// Gaussian Elimination with 10 rhs
    	versionrun[4][rep]=test_IMe_DGESV(versionname[4], verbose, rows, cols, 1);					// IMe with 1 rhs
    	versionrun[5][rep]=test_IMe_DGESV(versionname[5], verbose, rows, cols, nRHS);				// IMe with 10 rhs

		//////////////////////////////////////////////////////////////////////////////////

		for (i=0; i<versions; i++)
		{
			versiontot[i] += versionrun[i][rep];
			if (verbose>0)
			{
				printf("\n%s    call    run time: %f clk", versionname[i], versionrun[i][rep]);
			}
		}
    }
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
    return(0);
}
