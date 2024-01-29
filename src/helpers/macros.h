/*
 * macros.h
 *
 *  Created on: Apr 1, 2016
 *      Author: marcello
 */

#ifndef __IME_MACROS_H__
#define __IME_MACROS_H__

// https://www.geeksforgeeks.org/branch-prediction-macros-in-gcc/
#define likely(x)	 __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define STR(x) #x
#define TOSTR(x) STR(x)

#define CONCAT(a, b) a##b
#define FUNCNAME(a, b) CONCAT(a, b)

#define MAX(a, b) (((a) > (b)) ? (a) : (b)) 
#define MIN(a, b) (((a) < (b)) ? (a) : (b)) 

#define NULLFREE(x) if (x!=NULL) {free(x); x = NULL;}

#endif
