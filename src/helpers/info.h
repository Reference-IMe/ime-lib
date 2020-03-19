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

typedef struct result_info
{
	time_t total_start_time;
	time_t total_end_time;
	time_t core_start_time;
	time_t core_end_time;
	int	   exit_code;
	double norm_rel_err;
} result_info;

typedef struct run_info
{
	double total_time;
	double core_time;
	int	   exit_code;
	double norm_rel_err;
} run_info;


#endif
