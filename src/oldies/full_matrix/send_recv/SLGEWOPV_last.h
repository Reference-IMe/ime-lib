#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include <mpi.h>


void SLGEWOPV_calc_last(double** A, double* b, double** T, double* x, int n, double* h, double* hh, int rank, int nprocs)
{
    int i,j,l;

    int* map;
    map=malloc(2*n*sizeof(int));

    MPI_Bcast (&b[0],n,MPI_DOUBLE,0,MPI_COMM_WORLD);

	MPI_Datatype single_column;
	MPI_Type_vector (n , 1, 2*n , MPI_DOUBLE , & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/nprocs , 1, nprocs , MPI_DOUBLE , & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	for(i=0; i<2*n; i++)
	{
		map[i]= i % nprocs;
	}
	for(i=0; i<n; i++)
	{
		x[i]=0.0;
	}

	// init inhibition table
	if (rank==0)
	{
		for (j=0;j<n-1;j++) // all but last column
		{
			for (i=0;i<n;i++)
			{
				T[i][j+n]=A[j][i]/A[i][i];
				T[i][j]=0;
			}
			T[j][j]=1/A[j][j];
			if (map[j]!=0)
			{
				// send X and K parts
				// TODO: combine in a single Send
				MPI_Send (&T[0][j],1,single_column, map[j],j, MPI_COMM_WORLD);
				MPI_Send (&T[0][j+n],1,single_column, map[j],j+n, MPI_COMM_WORLD);
			}
		}
		//j=n-1; // last column
		for (i=0;i<n;i++)
		{
			T[i][j+n]=A[j][i]/A[i][i];
			T[i][j]=0;
		}
		T[j][j]=1/A[j][j];
		// send only X part
		MPI_Send (&T[0][n-1],1,single_column, map[n-1],n-1, MPI_COMM_WORLD);
	}
	else
	{
		// receive
		for (j=0;j<n-1;j++)
		{
			if (rank==map[j])
			{
				// TODO: combine in a single Recv
				MPI_Recv (&T[0][j],1,single_column, 0,j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv (&T[0][j+n],1,single_column, 0,j+n, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		//j=n-1
		if (rank==map[j])
		{
			MPI_Recv (&T[0][n-1],1,single_column, 0,n-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}

	// broadcast of the last col of T (K part)
	MPI_Bcast (&T[0][n*2-1],1,single_column,0,MPI_COMM_WORLD);

	// every node broadcasts its chunks of the last row of T (K part)
	for (j=0; j<nprocs; j++)
	{
		MPI_Bcast (&T[n-1][n+j],1,interleaved_row,j,MPI_COMM_WORLD);
	}

	// all levels but last one (l=0)
	for (l=n-1; l>0; l--)
	{
		for (i=l; i<=n-1; i++)
		{
			if(map[i]==rank)
			{
				x[i]=x[i]+T[l][i]*b[l];
			}
		}
		for (i=0; i<=l-1; i++)
		{
			b[i]=b[i]-T[l][n+i]*b[l];
			*h   =1/(1-T[i][n+l]*T[l][n+i]);
			*hh  =T[i][n+l]*(*h);

			/*
			if (rank==map[i])
			{
				T[i][i]=T[i][i]*(*h);
			}
			if (rank==map[l])
			{
				T[i][l]= -T[l][l]*(*hh);
			}
			*/

			// at every node: better calc than putting an IF ?
			T[i][i]=T[i][i]*(*h);
			T[i][l]= -T[l][l]*(*hh);
			//

			for (j=l+1; j<=n+l-1; j++)
			{
				if (rank==map[j])
				{
					T[i][j]=T[i][j]*(*h)-T[l][j]*(*hh);
				}
			}
		}

		// send chunks of K to "future" last node
		if(rank!=map[l-1])
		{
			MPI_Send (&T[l-1][n+rank],1,interleaved_row, map[l-1],rank, MPI_COMM_WORLD);
		}
		else // it's future last node running
		{
			for(j=0;j<nprocs;j++)
			{
				if(j!=rank)
				{
					MPI_Recv (&T[l-1][n+j],1,interleaved_row, j,j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
			}
		}
		//future last node broadcasts last row and col of K
		MPI_Bcast (&T[0][n+l-1],1,single_column,map[l-1],MPI_COMM_WORLD);
		MPI_Bcast (&T[l-1][n],n,MPI_DOUBLE,map[l-1],MPI_COMM_WORLD);
	}

	// last level (l=0)
	for (i=0; i<=n-1; i++)
	{
		if(map[i]==rank)
		{
			x[i]=x[i]+T[0][i]*b[0];
		}
	}

	// TODO: use MPI_Gather to
	// collect final solution
	if (rank!=0)
	{
		MPI_Send (&x[rank],1,interleaved_row, 0, rank, MPI_COMM_WORLD);
	}
	else
	{
		for(j=1;j<nprocs;j++)
		{
			MPI_Recv (&x[j],1,interleaved_row, j,j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}
}
