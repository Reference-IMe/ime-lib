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

#define NULLFREE(x) if (x!=NULL) {free(x); x = NULL;}
/*
 * calc process run time from start/end time
 * get maximum run time as process team run time
 * get exit code and normwise relative error
 */
#define TEST_END(routine_info, process_info, team_info)	(process_info).total_time = (double)((routine_info).total_end_time - (routine_info).total_start_time);\
														(process_info).core_time  = (double)((routine_info).core_end_time  - (routine_info).core_start_time); \
														MPI_Reduce( &(process_info.total_time), &(team_info.total_time), 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );\
														MPI_Reduce( &(process_info.core_time),  &(team_info.core_time),  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );\
														team_info.exit_code    = routine_info.exit_code;    \
														team_info.norm_rel_err = process_info.norm_rel_err; \
														MPI_Barrier(MPI_COMM_WORLD);\
											return team_info;
#endif
