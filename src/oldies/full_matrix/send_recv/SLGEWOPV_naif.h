#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include <mpi.h>


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// very old version below!
// don't use: may be buggy

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SLGEWOPV_calc_naif(double** A, double* b, double* s, int n, double** K, double* H, double* F, int rank, int nprocs)
{
    int i,j,l;
    int rows=n;
    int cols=n;
    double** X=A;
    double tmpAdiag;
    int* map;
    map=malloc(n*sizeof(int));

    MPI_Bcast (&A[0][0],n*n,MPI_DOUBLE,0,MPI_COMM_WORLD);

	for(i=0; i<n; i++)
	{
		map[i]= i % nprocs;
	}

/*	if(rank==0)
	{
		printf("mappings:\n");
		for (i=0;i<rows;i++)
		{
			printf("%d: #%d\n",i,map[i]);
		}
		printf("\n");
	}
*/
	for (i=0;i<rows;i++)
	{
		tmpAdiag=1/A[i][i];
		for (j=0;j<cols;j++)
		{
			if (i==j)
			{
				X[i][j]=tmpAdiag;
				K[i][j]=1;
			}
			else
			{
				K[i][j]=A[j][i]*tmpAdiag;

				// ATTENTION : transposed
				X[j][i]=0.0;

			}
		}
		F[i]=b[i];
		s[i]=0.0;
	}

	MPI_Datatype single_column;
	MPI_Type_vector (n , 1, n , MPI_DOUBLE , & single_column );
	MPI_Type_commit (& single_column);

	for (l=rows-1; l>0; l--)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		//printf("#%d doing level %d\n",rank,l);
		//printf("#%d received: ",rank);
		MPI_Barrier(MPI_COMM_WORLD);
		/*
		for (i=0;i<rows;i++)
		{
			printf("%f ",K[i][l]);
		}
		printf("\n");
		*/
		for (i=0; i<l; i++)
		{
			H[i]=1/(1-K[i][l]*K[l][i]);
			for (j=0; j<cols; j++)
			{
				if (map[j]==rank)
				{
					X[i][j]=H[i]*(X[i][j]-K[i][l]*X[l][j]);
				}
			}
			for (j=0; j<l; j++)
			{
				if (map[j]==rank)
				{
					//printf("#%d,%d calc %d.\n",rank,l,j);
					K[i][j]=H[i]*(K[i][j]-K[i][l]*K[l][j]);
				}
			}
		}
		for (j=0; j<l-1; j++)
		{
			MPI_Bcast (&K[l-1][j],1,MPI_DOUBLE,map[j],MPI_COMM_WORLD);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		//printf("#%d at barrier\n",rank);
		//MPI_Barrier(MPI_COMM_WORLD);
		/*
		if(rank==map[l-1])
		{
			printf("#%d,%d sending %d: ",map[l-1],l,l-1);
			for (i=0;i<rows;i++)
			{
				printf("%f ",K[i][l-1]);
			}
			printf("\n");
		}
		*/
		MPI_Bcast (&K[0][l-1],1,single_column,map[l-1],MPI_COMM_WORLD);
	}

//	MPI_Bcast (&K[0][0],1,single_column,map[0],MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

    if(rank==0)
    {
    	//PrintMatrix2D(K,rows,cols);
    	//printf("\n");
		for (i=rows-2; i>=0; i--)
		{
			for (l=i+1; l<rows; l++)
			{
				F[i]=F[i]-F[l]*K[l][i];
			}
		}
		//PrintVector(F,rows);
		//printf("\n");
    }
	MPI_Bcast (F,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	for (j=0; j<cols; j++)
	{
		if(rank==0)
		{
			if(map[j]!=rank)
			{
				MPI_Recv(&s[j],1, MPI_DOUBLE, map[j], j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			else
			{
				for (l=0; l<rows; l++)
				{
					s[j]=F[l]*X[l][j]+s[j];
				}
			}
		}
		else // rank!=0
		{
			if (map[j]==rank)
			{
				for (l=0; l<rows; l++)
				{
				s[j]=F[l]*X[l][j]+s[j];
				//printf("s[%d] = %f\n",j,s[j]);
				//printf("#%d sends %d\n",rank,j);
				}
				MPI_Send(&s[j],1, MPI_DOUBLE, 0, j, MPI_COMM_WORLD);
			}
		}
	}
}
