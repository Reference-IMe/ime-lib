/*
 * macros.h
 *
 *  Created on: Apr 1, 2016
 *      Author: marcello
 */

#ifndef __MACROS_H__
#define __MACROS_H__

#define MAX(a, b) (((a) > (b)) ? (a) : (b)) 
#define MIN(a, b) (((a) < (b)) ? (a) : (b)) 

#define TEST_END(info, timing, timing_max)	(timing).total = (double)((info).total_end_time - (info).total_start_time);\
											(timing).core = (double)((info).core_end_time - (info).core_start_time);\
											MPI_Reduce( &(timing).total, &(timing_max).total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );\
											MPI_Reduce( &(timing).core, &(timing_max).core, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );\
											MPI_Barrier(MPI_COMM_WORLD);\
											return (timing_max);


#endif
