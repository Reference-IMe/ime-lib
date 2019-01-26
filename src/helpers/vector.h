/*
 * vector.c
 *
 *  Created on: Aug 27, 2015
 *      Author: marcello
 */

#include <stdio.h>
#include <stdlib.h>
#include "types.h"

#ifndef __VECTOR_H__
#define __VECTOR_H__

double* AllocateVector(cui rows)
{
	return(malloc(rows*sizeof(double)));
}

void DeallocateVector(double* vec)
{
	free(vec);
}

void PrintVector(double* const vec, cui rows)
{
	ui r;
	for(r=0;r<rows;r++)
		printf("%g\n",vec[r]);
	//printf("\n");
}

void FillVector(double* vec, cui rows, double val)
{
	ui r;
	for(r=0;r<rows;r++)
		vec[r]=val;
}

#endif
