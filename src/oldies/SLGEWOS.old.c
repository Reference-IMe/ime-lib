#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../../../helpers/selfie.h"
#include "../../../helpers/matrix.h"

int main(int argc, char **argv)
{
	metrics program;

    int i,j,k,l,rep;
    double A[5000][5000],b[5000];
	double c[5000],x[5000],sum=0.0;
	/*
    double A[4][4] = {
    		{2,1,1,-1},
			{1,2,1,2},
			{1,1,3,2},
			{1,2,2,3}
    };
    double b[4]={1,1,1,1};
    */
    double X[5000][5000],K[5000][5000],H[5000],F[5000],S[5000];
    double GErunt[100];
    double LUt[100];
    double BSt[100];
    double IMt[100];
    double GEtotrunt=0.0;
    double IMtotrunt=0.0;

    int n=atoi(argv[1]);
    //int n=4;
    int rows=n;
    int cols=n;
    int ft=atoi(argv[2]);
    int repetitions=atoi(argv[3]);

    clock_t begin1, end1, begin2, end2, begin3, end3;
    time_t start1, stop1, start2, stop2, start3, stop3;

    for (rep=0; rep<repetitions; rep++)
    {
		printf("\n\n Run: %d",rep);

		for (i=0;i<rows;i++)
		{
			for (j=0;j<cols;j++)
			{
				if (j>=(rows-i))
					A[i][j]=2.;
				else
					A[i][j]=1.;
				if(i==j)
					A[i][j]++;
			}
			b[i]=1.;
		}
		A[0][cols-1]=-1.;

		/*
		printf("\n\n Matrix A:\n");
		for (i=0; i<n; i++)
		{
			for (j=0; j<n; j++)
				printf("%9.6lf ",A[i][j]);
			printf("\n");
		}
		printf("\n Vector b:\n");
		for (i=0; i<n; i++)
			printf("%9.6lf ",b[i]);
		printf("\n");
		*/



		//////////////////////////////////////////////////////////////////////////////////

		//begin1 =clock();
		getmetrics(&program);
		start1=program.wall_clock;

		for(k=0;k<n;k++)
		{
			for(i= k+1; i<n; i++)
			{
				c[i]=A[i][k]/A[k][k];
			}
			for(i= k+1; i<n; i++)
			{
				for(j=0;j<n;j++)
				{
					A[i][j]=A[i][j]-( c[i]*A[k][j] );
				}
				b[i]=b[i]-( c[i]*b[k] );
			}
		}
		//end1 = clock();
		getmetrics(&program);
		stop1=program.wall_clock;

		//////////////////////////////////////////////////////////////////////////////////

		//begin2 =clock();
		getmetrics(&program);
		start2=program.wall_clock;

		x[n-1]=b[n-1]/A[n-1][n-1];
		for(i=n-2;i>=0;i--)
		{
			sum=0;

			for(j=i+1;j<n;j++)
			{
				sum=sum+A[i][j]*x[j];
			}
			x[i]=(b[i]-sum)/A[i][i];
		}

		//end2 = clock();
		getmetrics(&program);
		stop2=program.wall_clock;

		//////////////////////////////////////////////////////////////////////////////////

		/*
		printf("\nThe GE solution is:");
		for(i=0;i<n;i++)
		{
			printf("\nx%d=%f\t",i,x[i]);

		}
		*/

		//////////////////////////////////////////////////////////////////////////////////

		for (i=0;i<rows;i++)
		{
			for (j=0;j<cols;j++)
			{
				if (j>=(rows-i))
					A[i][j]=2.;
				else
					A[i][j]=1.;
				if(i==j)
					A[i][j]++;
			}
			b[i]=1.;
		}
		A[0][cols-1]=-1.;

		/*/printf("\n\n Matrix A:\n");
		for (i=0; i<n; i++)
		{
			for (j=0; j<n; j++)
				printf("%9.6lf ",A[i][j]);
			printf("\n");
		}
		printf("\n Vector b:\n");
		for (i=0; i<n; i++)
			printf("%9.6lf ",b[i]);
		printf("\n");
		*/


		//begin3 = clock();
		getmetrics(&program);
		start3=program.wall_clock;

		for (i=0;i<rows;i++)
		{
			for (j=0;j<cols;j++)
			{
				if (i==j)
				{
					X[i][j]=1/A[i][j];
					//K[i][j]=1;
				}
				else
				{
					X[i][j]=0.;
					K[i][j]=A[j][i]/A[i][i];
				}
			}
			F[i]=b[i];
			S[i]=0.;
		}

		for (l=rows-1;l>0;l--)
		{
			for (i=0;i<l;i++)
			{
				H[i]=1/(1-K[i][l]*K[l][i]);
				for (j=0;j<cols;j++)
				{
					X[i][j]=H[i]*(X[i][j]-K[i][l]*X[l][j]);
				}
				for (j=0;j<l;j++)
				{
					if (j!=i)
					{
						K[i][j]=H[i]*(K[i][j]-K[i][l]*K[l][j]);
					}
				}
			}
		}

		/*
		printf("\nThe IMe principal solution is:");
		for(i=0;i<cols;i++)
		{
			printf("\nx%d=%f\t",i,X[0][i]);

		}
		*/

		for (i=rows-2;i>=0;i--)
		{
			for (l=i+1;l<rows;l++)
			{
				F[i]=F[i]-F[l]*K[l][i];
			}
		}

		for (j=0;j<cols;j++)
		{
			for (l=0;l<rows;l++)
			{
				S[j]=F[l]*X[l][j]+S[j];
			}
		}

		//end3 = clock();
		getmetrics(&program);
		stop3=program.wall_clock;

		//////////////////////////////////////////////////////////////////////////////////


		printf("\nThe IMe solution is:");
		for(i=0;i<cols;i++)
		{
			printf("\nx%d=%f\t",i,S[i]);

		}


		//////////////////////////////////////////////////////////////////////////////////
/*
		LUt[rep]=(double)(end1 - begin1) / CLOCKS_PER_SEC;
		BSt[rep]=(double)(end2 - begin2) / CLOCKS_PER_SEC;
		IMt[rep]=(double)(end3 - begin3) / CLOCKS_PER_SEC;
*/
		LUt[rep]=(double)(stop1 - start1);
		BSt[rep]=(double)(stop2 - start2);
		IMt[rep]=(double)(stop3 - start3);

		GErunt[rep]=LUt[rep]+BSt[rep];
		GEtotrunt += GErunt[rep];
		IMtotrunt += IMt[rep];

		printf("\n\nLU  decomposition time: %f", LUt[rep]);
		printf("\nBack substitution time: %f", BSt[rep]);
		printf("\nGaussian elimin.  time: %f\n", GErunt[rep]);
		printf("\nInhibition seq.   time: %f\n", IMt[rep]);
    }

	printf("\nGE Total   run time: %f", GEtotrunt);
	printf("\nGE Average run time: %f\n", GEtotrunt/repetitions);

	printf("\nIM Total   run time: %f", IMtotrunt);
	printf("\nIM Average run time: %f\n", IMtotrunt/repetitions);

    return(0);
}
