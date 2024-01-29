/*
 * matrix_basic.h
 *
 *  Created on: Aug 27, 2015
 *      Author: marcello
 */

#ifndef __MATRIX_BASIC_H__
#define __MATRIX_BASIC_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "types.h"
#include "macros.h"

//allocation types:
#define NONCONTIGUOUS 0
#define CONTIGUOUS 1

#define PRECISIONTYPE double
#include "matrix_basic_generic_precision.inc"
#undef PRECISIONTYPE

#define PRECISIONTYPE float
#include "matrix_basic_generic_precision.inc"
#undef PRECISIONTYPE

#endif
