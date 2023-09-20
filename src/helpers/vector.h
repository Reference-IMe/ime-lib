/*
 * vector.h
 *
 *  Created on: Aug 27, 2015
 *      Author: marcello
 */

#include <stdio.h>
#include <stdlib.h>
#include "macros.h"

#ifndef __VECTOR_H__
#define __VECTOR_H__

#define PRECISIONTYPE double
#include "vector_generic_precision.inc"
#undef PRECISIONTYPE

#define PRECISIONTYPE float
#include "vector_generic_precision.inc"
#undef PRECISIONTYPE

/*
double* AllocateVector_double(int rows)
{
	return malloc(rows*sizeof(double));
}

void DeallocateVector_double(double* vec)
{
	NULLFREE(vec);
}

void PrintVector_double(double* const vec, int rows)
{
	int r;
	for(r=0;r<rows;r++)
		printf("%.3g\n",vec[r]);
	//printf("\n");
}

void FillVector_double(double* vec, int rows, double val)
{
	int r;
	for(r=0;r<rows;r++)
		vec[r]=val;
}
*/

void PrintVector_int(int* const vec, int rows)
{
	int r;
	for(r=0;r<rows;r++)
		printf("%d\n",vec[r]);
	//printf("\n");
}

#endif
