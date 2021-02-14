/*
 * pbDGEIT_CX.macros.h
 *
 *  Created on: Feb 14, 2021
 *      Author: marcello
 */

#ifndef __PBDGEIT_CX_MACROS_H__
#define __PBDGEIT_CX_MACROS_H__


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
