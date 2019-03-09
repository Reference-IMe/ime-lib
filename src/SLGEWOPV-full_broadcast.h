#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers/selfie.h"
#include "helpers/matrix.h"
#include "helpers/vector.h"
#include <mpi.h>

void SLGEWOPV_calc_last(double** A, double* b, double** T, double* x, int n, double* h, double* hh, int rank, int nprocs)
{
    int i,j,l;

    int* map;
    map=malloc(n*sizeof(int));

    MPI_Bcast (&A[0][0],n*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
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
	}

	for (i=0;i<n;i++)
	{
		for (j=0;j<n;j++)
		{
			T[i][j+n]=A[j][i]/A[i][i];
			T[i][j]=0;
		}
		T[i][i]=1/A[i][i];
		x[i]=0.0;
	}

	// all levels but last (l=0)
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
			T[i][i]=T[i][i]*(*h);
			//T[i][l]= -T[l][l]*T[i][n+l]*H[i];
			T[i][l]= -T[l][l]*(*hh);
			for (j=l+1; j<=n+l-1; j++)
			{
				//T[i][j]=(T[i][j]-T[l][j]*T[i][n+l])*H[i];
				T[i][j]=T[i][j]*(*h)-T[l][j]*(*hh);
			}
		}

		if(rank!=map[l-1])
		{
			MPI_Send (&T[l-1][n+rank],1,interleaved_row, map[l-1],rank, MPI_COMM_WORLD);
		}
		else
		{
			for(j=0;j<nprocs;j++)
			{
				if(j!=rank)
				{
					MPI_Recv (&T[l-1][n+j],1,interleaved_row, j,j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
			}
		}
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


/*
 * IF in init removed
 * IF in calc not completely removed
 * send chunks of last row to last node
 * broadcast last row and last col
 * inefficiently broadcast each x[i]
 */
void SLGEWOPV_calc_last_unopt(double** A, double* b, double** T, double* x, int n, double* h, double* hh, int rank, int nprocs)
{
    int i,j,l;
    //int rows=n;
    //int cols=n;

    //double h,hh;
    int* map;
    map=malloc(n*sizeof(int));

    MPI_Bcast (&A[0][0],n*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
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
	}

	for (i=0;i<n;i++)
	{
		for (j=0;j<n;j++)
		{
			T[i][j+n]=A[j][i]/A[i][i];
			T[i][j]=0;
		}
		T[i][i]=1/A[i][i];
		x[i]=0.0;
	}

	for (l=n-1; l>=0; l--)
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
			T[i][i]=T[i][i]*(*h);
			//T[i][l]= -T[l][l]*T[i][n+l]*H[i];
			T[i][l]= -T[l][l]*(*hh);
			for (j=l+1; j<=n+l-1; j++)
			{
				//T[i][j]=(T[i][j]-T[l][j]*T[i][n+l])*H[i];
				T[i][j]=T[i][j]*(*h)-T[l][j]*(*hh);
			}
		}

		if(rank!=map[l-1])
		{
			if(l>0) MPI_Send (&T[l-1][n+rank],1,interleaved_row, map[l-1],rank, MPI_COMM_WORLD);
		}
		else
		{
			for(j=0;j<nprocs;j++)
			{
				if(j!=rank)
				{
					if(l>0) MPI_Recv (&T[l-1][n+j],1,interleaved_row, j,j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
			}
		}
		if(l>0)
		{
			MPI_Bcast (&T[0][n+l-1],1,single_column,map[l-1],MPI_COMM_WORLD);
			MPI_Bcast (&T[l-1][n],n,MPI_DOUBLE,map[l-1],MPI_COMM_WORLD);
		}
		else
		{
			for (i=0;i<n;i++)
				{
					MPI_Bcast (&x[i],1,MPI_DOUBLE,map[i],MPI_COMM_WORLD);
				}
		}
	}
}


/*
 * IF in init not removed
 * IF in calc removed
 * send chunks of last row to last node
 * broadcast last row and last col
 * master calcs and broadcasts auxiliary causes
 */
void SLGEWOPV_calc_sendopt(double** A, double* b, double* s, int n, double** K, double* H, int rank, int nprocs)
{
    int i,j,l;
    int rows=n;
    int cols=n;
    double** X=A;
    double*  F=b;
    double tmpAdiag;
    int* map;
    map=malloc(n*sizeof(int));

    MPI_Bcast (&A[0][0],n*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast (&b[0],n,MPI_DOUBLE,0,MPI_COMM_WORLD);

	for(i=0; i<n; i++)
	{
		map[i]= i % nprocs;
	}

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
		s[i]=0.0;
	}

	MPI_Datatype single_column;
	MPI_Type_vector (n , 1, n , MPI_DOUBLE , & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/nprocs , 1, nprocs , MPI_DOUBLE , & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	for (l=rows-1; l>0; l--)
	{
		for (i=0; i<l; i++)
		{
			H[i]=1/(1-K[i][l]*K[l][i]);
		}
		for (j=rank; j<l; j+=nprocs)
		{
			X[j][j]=H[j]*(X[j][j]-K[j][l]*X[l][j]);
		}
		for (i=0; i<l; i++)
		{

			for (j=cols-(nprocs-rank-1)-1; j>=l; j-=nprocs)
			{
				X[i][j]=H[i]*(X[i][j]-K[i][l]*X[l][j]);
			}
			for (j=rank; j<l; j+=nprocs)
			{
				K[i][j]=H[i]*(K[i][j]-K[i][l]*K[l][j]);
			}
		}
		if(rank!=map[l-1])
		{
			MPI_Send (&K[l-1][rank],1,interleaved_row, map[l-1],rank, MPI_COMM_WORLD);
		}
		else
		{
			for(j=0;j<nprocs;j++)
			{
				if(j!=rank)
				{
					MPI_Recv (&K[l-1][j],1,interleaved_row, j,j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
			}
		}

		MPI_Bcast (&K[0][l-1],1,single_column,map[l-1],MPI_COMM_WORLD);
		MPI_Bcast (&K[l-1][0],cols,MPI_DOUBLE,map[l-1],MPI_COMM_WORLD);
	}

    if(rank==0)
    {
		for (i=rows-2; i>=0; i--)
		{
			for (l=i+1; l<rows; l++)
			{
				F[i]=F[i]-F[l]*K[l][i];
			}
		}
    }

	MPI_Bcast (F,n,MPI_DOUBLE,0,MPI_COMM_WORLD);

	for (j=rank; j<cols; j+=nprocs)
	{
		for (l=0; l<rows; l++)
		{
			s[j]=F[l]*X[l][j]+s[j];
		}
	}
	for (j=0; j<nprocs; j++)
	{
		MPI_Bcast (&s[j],1,interleaved_row,j,MPI_COMM_WORLD); // too much! avoid broadcast! a send is enough
	}
}


/*
 * IF in init not removed
 * IF in calc removed
 * every node broadcasts chunks of last row
 * last node broadcast last col
 * master calcs and broadcasts auxiliary causes
 */
void SLGEWOPV_calc_unswitch(double** A, double* b, double* s, int n, double** K, double* H, int rank, int nprocs)
{
    int i,j,l;
    int rows=n;
    int cols=n;
    double** X=A;
    double*  F=b;
    double tmpAdiag;
    int* map;
    map=malloc(n*sizeof(int));

    MPI_Bcast (&A[0][0],n*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast (&b[0],n,MPI_DOUBLE,0,MPI_COMM_WORLD);

	for(i=0; i<n; i++)
	{
		map[i]= i % nprocs;
	}

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
		s[i]=0.0;
	}

	MPI_Datatype single_column;
	MPI_Type_vector (n , 1, n , MPI_DOUBLE , & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/nprocs , 1, nprocs , MPI_DOUBLE , & interleaved_row );
	MPI_Type_commit (& interleaved_row);

	for (l=rows-1; l>0; l--)
	{
		for (i=0; i<l; i++)
		{
			H[i]=1/(1-K[i][l]*K[l][i]);
		}
		for (j=rank; j<l; j+=nprocs)
		{
			X[j][j]=H[j]*(X[j][j]-K[j][l]*X[l][j]);
		}
		for (i=0; i<l; i++)
		{

			for (j=cols-(nprocs-rank-1)-1; j>=l; j-=nprocs)
			{
				X[i][j]=H[i]*(X[i][j]-K[i][l]*X[l][j]);
			}
			for (j=rank; j<l; j+=nprocs)
			{
				K[i][j]=H[i]*(K[i][j]-K[i][l]*K[l][j]);
			}
		}

		for (j=0; j<nprocs; j++)
		{
			MPI_Bcast (&K[l-1][j],1,interleaved_row,j,MPI_COMM_WORLD);
		}

		MPI_Bcast (&K[0][l-1],1,single_column,map[l-1],MPI_COMM_WORLD);
	}

    if(rank==0)
    {
		for (i=rows-2; i>=0; i--)
		{
			for (l=i+1; l<rows; l++)
			{
				F[i]=F[i]-F[l]*K[l][i];
			}
		}
    }

	MPI_Bcast (F,n,MPI_DOUBLE,0,MPI_COMM_WORLD);

	for (j=rank; j<cols; j+=nprocs)
	{
		for (l=0; l<rows; l++)
		{
			s[j]=F[l]*X[l][j]+s[j];
		}
	}
	for (j=0; j<nprocs; j++)
	{
		MPI_Bcast (&s[j],1,interleaved_row,j,MPI_COMM_WORLD);
	}
}


/*
 * IF in init not removed
 * IF in calc not removed
 * send chunks of last row to last node
 * broadcast last row and last col
 * every node calcs auxiliary causes
 * calc and broadcast solution
 */
void SLGEWOPV_calc(double** A, double* b, double* s, int n, double** K, double* H, int rank, int nprocs)
{
    int i,j,l;
    int rows=n;
    int cols=n;
    double** X=A;
    double*  F=b;
    double tmpAdiag;
    int* map;
    map=malloc(n*sizeof(int));

    MPI_Bcast (&A[0][0],n*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast (&b[0],n,MPI_DOUBLE,0,MPI_COMM_WORLD);

	for(i=0; i<n; i++)
	{
		map[i]= i % nprocs;
	}

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
		s[i]=0.0;
	}

	MPI_Datatype single_column;
	MPI_Type_vector (n , 1, n , MPI_DOUBLE , & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/nprocs , 1, nprocs , MPI_DOUBLE , & interleaved_row );
	MPI_Type_commit (& interleaved_row);

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

		for (j=0; j<nprocs; j++)
		{
			MPI_Bcast (&K[l-1][j],1,interleaved_row,j,MPI_COMM_WORLD);
		}

		MPI_Bcast (&K[0][l-1],1,single_column,map[l-1],MPI_COMM_WORLD);

		for (i=l-1; i>=0; i--)
		{
				F[i]=F[i]-F[l]*K[l][i];
		}
	}
/*
    if(rank==0)
    {
		for (i=rows-2; i>=0; i--)
		{
			for (l=i+1; l<rows; l++)
			{
				F[i]=F[i]-F[l]*K[l][i];
			}
		}
    }

	MPI_Bcast (F,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
*/
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
		MPI_Bcast (&s[j],1,interleaved_row,j,MPI_COMM_WORLD); // too much! avoid broadcast! a send is enough
	}
}

void SLGEWOPV_calc_F(double** A, double* b, double* s, int n, double** K, double* H, int rank, int nprocs)
{
    int i,j,l;
    int rows=n;
    int cols=n;
    double** X=A;
    double*  F=b;
    double tmpAdiag;
    int* map;
    map=malloc(n*sizeof(int));

    MPI_Bcast (&A[0][0],n*n,MPI_DOUBLE,0,MPI_COMM_WORLD);

	for(i=0; i<n; i++)
	{
		map[i]= i % nprocs;
	}

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
		s[i]=0.0;
	}

	MPI_Datatype single_column;
	MPI_Type_vector (n , 1, n , MPI_DOUBLE , & single_column );
	MPI_Type_commit (& single_column);

	MPI_Datatype interleaved_row;
	MPI_Type_vector (n/nprocs , 1, nprocs , MPI_DOUBLE , & interleaved_row );
	MPI_Type_commit (& interleaved_row);

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

		for (j=0; j<nprocs; j++)
		{
			MPI_Bcast (&K[l-1][j],1,interleaved_row,j,MPI_COMM_WORLD);
		}

		MPI_Bcast (&K[0][l-1],1,single_column,map[l-1],MPI_COMM_WORLD);
	}

    if(rank==0)
    {
		for (i=rows-2; i>=0; i--)
		{
			for (l=i+1; l<rows; l++)
			{
				F[i]=F[i]-F[l]*K[l][i];
			}
		}
    }

	MPI_Bcast (F,n,MPI_DOUBLE,0,MPI_COMM_WORLD);

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
		MPI_Bcast (&s[j],1,interleaved_row,j,MPI_COMM_WORLD);
	}
}

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

void SLGEWOPV(double** A, double* b, double* s, int n, int rank, int nprocs)
{
    //double** X;
    double** K;
    double*  H;
    //double*  F;

    //X=AllocateMatrix2D(n,n,CONTIGUOUS);
    K=AllocateMatrix2D(n,n,CONTIGUOUS);

    H=AllocateVector(n);
    //F=AllocateVector(n);

    SLGEWOPV_calc_unswitch(A, b, s, n, K, H, rank, nprocs);
    //SLGEWOS_calc(A, b, s, n, X, K, H);

    //DeallocateMatrix2D(X,n,CONTIGUOUS);
    DeallocateMatrix2D(K,n,CONTIGUOUS);

    DeallocateVector(H);
    //DeallocateVector(F);
}

