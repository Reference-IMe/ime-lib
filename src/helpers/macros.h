/*
 * macros.h
 *
 *  Created on: Apr 1, 2016
 *      Author: marcello
 */

#ifndef __MACROS_H__
#define __MACROS_H__

// https://www.geeksforgeeks.org/branch-prediction-macros-in-gcc/
#define likely(x)	 __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

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

#define MAIN_CLEANUP(mpi_rank)	if (mpi_rank==0)								\
								{												\
									if (output_to_file || input_from_file)		\
									{											\
										if (fp != NULL) fclose(fp);				\
									}											\
									DeallocateMatrix1D(A_ref);					\
									DeallocateVector(b_ref);					\
									DeallocateVector(x_ref);					\
								}												\
								sdsfree(test_output_file_name);

#define PVLOCAL(col_index, num_of_cols) col_index-((col_index)/(num_of_cols))*(num_of_cols)
#define PVMAP(col_index, num_of_cols) ((col_index)/(num_of_cols))
#define PVGLOBAL(col_index, num_of_cols, mpi_rank) col_index+(num_of_cols)*(mpi_rank)

#define IS_MULT(a, b) ( (a % b)==0 )

#define IS_SQUARE(a) ( pow((int)sqrt(a),2)==a )

#define TAG2LABEL(tag_string, label_string) label_string=sdscat(label_string,tag_string);	\
											label=sdscat(label, ":");

#define DISPLAY_MSG(routine_name, text) printf("   > %-31s%s\n", routine_name, text);
#define DISPLAY_ERR(routine_name, text) printf("ERR> %-31s%s\n", routine_name, text);
#define DISPLAY_ERR_GLB(text)           printf("ERR> %s\n", text);
#define DISPLAY_WRN(routine_name, text) printf("WRN> %-31s%s\n", routine_name, text);
#define DISPLAY_WRN_GLB(text)           printf("WRN> %s\n", text);

#endif
