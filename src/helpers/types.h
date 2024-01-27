/*
 * types.h
 *
 *  Created on: Aug 27, 2015
 *      Author: marcello
 */

#ifndef __IME_TYPES_H__
#define __IME_TYPES_H__

typedef struct exit_status
{
	time_t total_start_time;
	time_t total_end_time;
	time_t core_start_time;
	time_t core_end_time;
	char   exit_code;
} exit_status;

#endif
