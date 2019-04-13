#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include <mpi.h>


/*
 * IF in init not removed
 * IF in calc not removed
 * send chunks of last row to last node
 * broadcast last row and last col
 * every node calcs auxiliary causes
 * calc and broadcast solution
 */
void SLGEWOPV_calc_base(double** A, double* b, double* s, int n, double** K, double* H, int rank, int nprocs)
{
    int i,j,l;
    int rows=n;
    int cols=n;
    double** X=A;
    double*  F=b;
    double tmpAdiag;
    int* map;
    map=malloc(n*sizeof(int));

    MPI_Bcast (&b[0],n,MPI_DOUBLE,0,MPI_COMM_WORLD);

	MPI_Datatype single_column;
	MPI_Type_vector (n , 1, n , MPI_DOUBLE , & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/nprocs , 1, nprocs , MPI_DOUBLE , & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	for(i=0; i<n; i++)
	{
		map[i]= i % nprocs;
		s[i]=0.0;
	}

	//init inhibition table
	if (rank==0)
	{
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
		}
	}

	// spread inhibition table
	if (rank==0)
	{
		for (j=0;j<cols-1;j++) // to last but one
		{
			// send columns to single nodes
			if (map[j]!=0)
			{
				MPI_Send (&X[0][j],1,single_column, map[j],j, MPI_COMM_WORLD);
				MPI_Send (&K[0][j],1,single_column, map[j],j+n, MPI_COMM_WORLD);
			}
		}
		// last one of K is broadcasted to all nodes later
		MPI_Send (&X[0][cols-1],1,single_column, map[cols-1],cols-1, MPI_COMM_WORLD);
	}
	else
	{
		for (j=0;j<cols-1;j++)
		{
			// receive
			if (rank==map[j])
			{
				MPI_Recv (&X[0][j],1,single_column, 0,j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv (&K[0][j],1,single_column, 0,j+n, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		if (rank==map[cols-1])
			{
			MPI_Recv (&X[0][cols-1],1,single_column, 0,cols-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
	}

	// broadcast last column of K
	MPI_Bcast (&K[0][n-1],1,single_column,0,MPI_COMM_WORLD);

	// every node broadcasts its chunks of the last row of K
	for (j=0; j<nprocs; j++)
	{
		MPI_Bcast (&K[n-1][j],1,interleaved_row,j,MPI_COMM_WORLD);
	}

	//calc inhibition sequence
	for (l=rows-1; l>0; l--)
	{
		for (i=0; i<l; i++)
		{
			H[i]=1/(1-K[i][l]*K[l][i]);
			if(map[i]==rank)
			{
				X[i][i]=H[i]*(X[i][i]);
			}
			for (j=l; j<cols; j++)
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
					K[i][j]=H[i]*(K[i][j]-K[i][l]*K[l][j]);
				}
			}
		}

		// every node broadcasts its chunks of the last row of K
		for (j=0; j<nprocs; j++)
		{
			MPI_Bcast (&K[l-1][j],1,interleaved_row,j,MPI_COMM_WORLD);
		}

		// broadcast last column of K
		MPI_Bcast (&K[0][l-1],1,single_column,map[l-1],MPI_COMM_WORLD);

		// calc auxiliary causes on every node
		for (i=l-1; i>=0; i--)
		{
				F[i]=F[i]-F[l]*K[l][i];
		}
	}

	// final solution
	for (j=0; j<cols; j++)
	{
		if (map[j]==rank)
		{
			for (l=0; l<rows; l++)
			{
				s[j]=F[l]*X[l][j]+s[j];
			}
		}
	}

	for (j=0; j<nprocs; j++)
	{
		MPI_Bcast (&s[j],1,interleaved_row,j,MPI_COMM_WORLD); // too much! avoid broadcasts! a send is enough
	}
}
