/*
 * matrix.h
 *
 *  Created on: Aug 27, 2015
 *      Author: marcello
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "types.h"

//allocation types:
#define NONCONTIGUOUS 0
#define CONTIGUOUS 1

#ifndef __MATRIX_H__
#define __MATRIX_H__

//// 2D matrices

double** AllocateMatrix2D(cui rows, cui cols, cui allocation_type)
{
	double** mat;
	ui r;

	if (allocation_type==NONCONTIGUOUS)
	{
		mat=malloc(rows*sizeof(double));
		for(r=0;r<rows;r++)
			mat[r]=malloc(cols*sizeof(double));
	}
	else
	{
		mat=malloc(rows*sizeof(double*));
		mat[0]=malloc(rows*cols*sizeof(double));
		for(r=1; r<rows; r++)
			mat[r]=mat[r-1]+cols;
	}
	return(mat);
}

#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
void DeallocateMatrix2D(double** mat, cui rows, cui allocation_type)
{
	ui r;

	if (allocation_type==NONCONTIGUOUS)
	{
		for(r=0;r<rows;r++)
			free(mat[r]);
		free(mat);
	}
	else
	{
		free(mat[0]);
		free(mat);
	}

}

void PrintMatrix2D(double** const mat, cui rows, cui cols)
{
	ui r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			printf("%.3g\t",mat[r][c]);
		}
		printf("\n");
	}
	//printf("\n");
}

void FillMatrix2D(double** mat, cui rows, cui cols)
{
	ui r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			//mat[r][c]=(double)(pow(r*rows+c+1,2));
			if(c>=(rows-r))
				mat[r][c]=2.;
			else
				mat[r][c]=1.;
			if(r==c)
				mat[r][c]++;
		}
	}
	mat[0][cols-1]=-1.;
}

void OneMatrix2D(double** mat, cui rows, cui cols)
{
	ui r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			mat[r][c]=1;
		}
	}
}

void ReferenceMatrix2D(double** mat, cui rows, cui cols)
{
	ui r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			mat[r][c]=pow(10,ceil(log10(cols+1)))*(r+1)+c+1;
		}
	}
}

//// 1D matrices

double* AllocateMatrix1D(cui rows, cui cols)
{
	return(malloc(rows*cols*sizeof(double)));
}

void DeallocateMatrix1D(double* mat)
{
	free(mat);
}

void PrintMatrix1D(double* const mat, cui rows, cui cols)
{
	ui r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			printf("%.3g\t",mat[r*cols+c]);
		}
		printf("\n");
	}
	//printf("\n");
}

void FillMatrix1D(double* mat, cui rows, cui cols)
{
	ui r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			//mat[r*rows+c]=(double)(pow(r*rows+c+1,2));

			if(c>=(rows-r))
				mat[r*cols+c]=2.;
			else
				mat[r*cols+c]=1.;
			if(r==c)
				mat[r*cols+c]++;
				
		}
	}
	mat[0+cols-1]=-1.;
}

void FillMatrixT1D(double* mat, cui rows, cui cols) // Transposed
{
	ui r,c;

	for(c=0;c<cols;c++)
	{
		for(r=0;r<rows;r++)
		{
			//mat[r+cols*c]=(double)(pow(r*rows+c+1,2));
			if(c>=(rows-r))
				mat[r+rows*c]=2.;
			else
				mat[r+rows*c]=1.;
			if(r==c)
				mat[r+rows*c]++;
		}
	}
	mat[0+rows*(cols-1)]=-1.;
}

void OneMatrix1D(double* mat, cui rows, cui cols)
{
	ui r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			mat[r*cols+c]=1;
		}
	}
}

#endif
