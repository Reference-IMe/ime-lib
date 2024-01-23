/*
 * vector.h
 *
 *  Created on: Aug 27, 2015
 *      Author: marcello
 */

PRECISIONTYPE* FUNCNAME(AllocateVector_, PRECISIONTYPE) (int rows)
{
	return malloc(rows*sizeof(PRECISIONTYPE));
}

void FUNCNAME(DeallocateVector_, PRECISIONTYPE) (PRECISIONTYPE* vec)
{
	NULLFREE(vec);
}

void FUNCNAME(PrintVector_, PRECISIONTYPE) (PRECISIONTYPE* const vec, int rows)
{
	int r;
	for(r=0;r<rows;r++)
		printf("%.3g\n",vec[r]);
	//printf("\n");
}

void FUNCNAME(FillVector_, PRECISIONTYPE) (PRECISIONTYPE* vec, int rows, PRECISIONTYPE val)
{
	int r;
	for(r=0;r<rows;r++)
		vec[r]=val;
}
