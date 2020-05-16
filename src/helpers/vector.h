/*
 * vector.h
 *
 *  Created on: Aug 27, 2015
 *      Author: marcello
 */

#include <stdio.h>
#include <stdlib.h>
//#include "types.h"

#ifndef __VECTOR_H__
#define __VECTOR_H__

double* AllocateVector(int rows)
{
	return(malloc(rows*sizeof(double)));
}

void DeallocateVector(double* vec)
{
	NULLFREE(vec);
}

void PrintVector(double* const vec, int rows)
{
	int r;
	for(r=0;r<rows;r++)
		printf("%.3g\n",vec[r]);
	//printf("\n");
}

void PrintVectorINT(int* const vec, int rows)
{
	int r;
	for(r=0;r<rows;r++)
		printf("%d\n",vec[r]);
	//printf("\n");
}

void FillVector(double* vec, int rows, double val)
{
	int r;
	for(r=0;r<rows;r++)
		vec[r]=val;
}

#endif
