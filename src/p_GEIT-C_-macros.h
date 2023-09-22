/*
 * p_GEIT_C_-macros.h
 *
 *  Created on: Feb 14, 2021
 *      Author: marcello
 */

#ifndef __PB_GEIT_CX_MACROS_H__
#define __PB_GEIT_CX_MACROS_H__

// "hash" parameter has to be # (without quotes) when calling
// (preprocessor directive are not allowed in #defines

#define INIT_T_ON_DIAG(i, j, myrows, mycols, diag)	for (i=0;i<myrows;i++)						\
													{											\
														for (j=0;j<i;j++)						\
														{										\
															Tlocal[i][j]=Tlocal[i][j]/diag[i];	\
														}										\
														Tlocal[i][i]=1/diag[i];					\
														for (j=i+1;j<mycols;j++)				\
														{										\
															Tlocal[i][j]=Tlocal[i][j]/diag[i];	\
														}										\
													}

#define INIT_T_OFF_DIAG(i, j, myrows, mycols, diag)	for (i=0;i<myrows;i++)						\
													{											\
														for (j=0;j<mycols;j++)					\
														{										\
															Tlocal[i][j]=Tlocal[i][j]/diag[i];	\
														}										\
													}

#endif
