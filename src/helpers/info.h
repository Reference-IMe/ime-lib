/*
 * timings.h
 *
 *  Created on: Feb 13, 2020
 *      Author: marcello
 *
 *      https://stackoverflow.com/questions/13156031/measuring-time-in-c
 */

#include <time.h>

#ifndef __INFO_H__
#define __INFO_H__

typedef struct test_input
{
	int n;
	double* A_ref;
	double* x_ref;
	double* b_ref;
	int nrhs;
	int calc_procs;
	int spare_procs;
	int ime_bf;
	int scalapack_bf;
} test_input;

typedef struct test_output
{
	time_t total_start_time;
	time_t total_end_time;
	time_t core_start_time;
	time_t core_end_time;
	int	   exit_code;
	double norm_rel_err;
} test_output;

typedef struct test_result
{
	double total_time;
	double core_time;
	int	   exit_code;
	double norm_rel_err;
} test_result;


#endif
