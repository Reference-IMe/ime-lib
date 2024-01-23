
PRECISIONTYPE** FUNCNAME(AllocateMatrix2D_, PRECISIONTYPE) (int rows, int cols, int allocation_type)
{
	PRECISIONTYPE** mat;
	int r;

	if (allocation_type==NONCONTIGUOUS)
	{
		mat=malloc(rows*sizeof(PRECISIONTYPE));
		for(r=0;r<rows;r++)
			mat[r]=malloc(cols*sizeof(PRECISIONTYPE));
	}
	else
	{
		mat=malloc(rows*sizeof(PRECISIONTYPE*));
		mat[0]=malloc(rows*cols*sizeof(PRECISIONTYPE));
		for(r=1; r<rows; r++)
			mat[r]=mat[r-1]+cols;
	}
	return mat;
}

//#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
void FUNCNAME(DeallocateMatrix2D_, PRECISIONTYPE) (PRECISIONTYPE** mat, int rows, int allocation_type)
{
	int r;

	if (mat!=NULL)
	{
		if (allocation_type==NONCONTIGUOUS)
		{
			for(r=0;r<rows;r++)
				NULLFREE(mat[r]);
			NULLFREE(mat);
		}
		else
		{
			NULLFREE(mat[0]);
			NULLFREE(mat);
		}
	}
}

void FUNCNAME(PrintMatrix2D_, PRECISIONTYPE) (PRECISIONTYPE** const mat, int rows, int cols)
{
	int r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			printf("%.3f\t",mat[r][c]);
		}
		printf("\n");
	}
	//printf("\n");
}

void FUNCNAME(FillMatrix2D_, PRECISIONTYPE) (PRECISIONTYPE** mat, int rows, int cols)
{
	int r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			//mat[r][c]=(PRECISIONTYPE)(pow(r*rows+c+1,2));
			if(c>=(rows-r))
				mat[r][c]=2.;
			else
				mat[r][c]=1.;
			if(r==c)
				mat[r][c]++;
		}
	}
	mat[0][cols-1]=-1.;
}

void FUNCNAME(SetMatrix2D_, PRECISIONTYPE) (PRECISIONTYPE setval, PRECISIONTYPE** mat, int rows, int cols)
{
	int r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			mat[r][c]=setval;
		}
	}
}

void FUNCNAME(OneMatrix2D_, PRECISIONTYPE) (PRECISIONTYPE** mat, int rows, int cols)
{
	FUNCNAME(SetMatrix2D_, PRECISIONTYPE)(1, mat, rows, cols);
}

void FUNCNAME(ReferenceMatrix2D_, PRECISIONTYPE) (PRECISIONTYPE** mat, int rows, int cols)
{
	int r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			mat[r][c]=pow(10,ceil(log10(cols+1)))*(r+1)+c+1;
		}
	}
}

void FUNCNAME(RandomMatrix2D_, PRECISIONTYPE) (PRECISIONTYPE** mat, int rows, int cols, int seed)
{
	// random generation in a given interval: https://stackoverflow.com/questions/13408990/how-to-generate-random-float-number-in-c

	int r,c;

	srand(seed);

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			mat[r][c]=((PRECISIONTYPE)rand()) / ((PRECISIONTYPE)RAND_MAX);
		}
	}
}

//// 1D matrices

PRECISIONTYPE* FUNCNAME(AllocateMatrix1D_, PRECISIONTYPE) (int rows, int cols)
{
	return malloc(rows*cols*sizeof(PRECISIONTYPE));
}

void FUNCNAME(DeallocateMatrix1D_, PRECISIONTYPE) (PRECISIONTYPE* mat)
{
	NULLFREE(mat);
}

void FUNCNAME(PrintMatrix1D_, PRECISIONTYPE) (PRECISIONTYPE* const mat, int rows, int cols)
{
	int r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			printf("%.3f\t",mat[r*cols+c]);
		}
		printf("\n");
	}
	//printf("\n");
}

void FUNCNAME(FillMatrix1D_, PRECISIONTYPE) (PRECISIONTYPE* mat, int rows, int cols)
{
	int r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			//mat[r*rows+c]=(PRECISIONTYPE)(pow(r*rows+c+1,2));

			if(c>=(rows-r))
				mat[r*cols+c]=2.;
			else
				mat[r*cols+c]=1.;
			if(r==c)
				mat[r*cols+c]++;
				
		}
	}
	mat[0+cols-1]=-1.;
}

void FUNCNAME(FillMatrixT1D_, PRECISIONTYPE) (PRECISIONTYPE* mat, int rows, int cols) // Transposed
{
	int r,c;

	for(c=0;c<cols;c++)
	{
		for(r=0;r<rows;r++)
		{
			//mat[r+cols*c]=(PRECISIONTYPE)(pow(r*rows+c+1,2));
			if(c>=(rows-r))
				mat[r+rows*c]=2.;
			else
				mat[r+rows*c]=1.;
			if(r==c)
				mat[r+rows*c]++;
		}
	}
	mat[0+rows*(cols-1)]=-1.;
}


void FUNCNAME(SetMatrix1D_, PRECISIONTYPE) (PRECISIONTYPE setval, PRECISIONTYPE* mat, int rows, int cols)
{
	int r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			mat[r*cols+c]=setval;
		}
	}
}

void FUNCNAME(OneMatrix1D_, PRECISIONTYPE) (PRECISIONTYPE* mat, int rows, int cols)
{
	FUNCNAME(SetMatrix1D_, PRECISIONTYPE)(1, mat, rows, cols);
}

void FUNCNAME(DiagonalMatrix1D_, PRECISIONTYPE) (PRECISIONTYPE diagval, PRECISIONTYPE offdiagval, PRECISIONTYPE* mat, int rows, int cols)
{
	int i,imin;

	imin = MIN(rows, cols);

	FUNCNAME(SetMatrix1D_, PRECISIONTYPE)(offdiagval, mat, rows, cols);

	for(i=0;i<imin;i++)
	{
			mat[i*cols+i]=diagval;
	}
}

void FUNCNAME(ReferenceMatrix1D_, PRECISIONTYPE) (PRECISIONTYPE* mat, int rows, int cols)
{
	int r,c;

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			mat[r*cols+c]=pow(10,ceil(log10(cols+1)))*(r+1)+c+1;
		}
	}
}

void FUNCNAME(RandomMatrix1D_, PRECISIONTYPE) (PRECISIONTYPE* mat, int rows, int cols, int seed)
{
	// random generation in a given interval: https://stackoverflow.com/questions/13408990/how-to-generate-random-float-number-in-c

	int r,c;

	srand(seed);

	for(r=0;r<rows;r++)
	{
		for(c=0;c<cols;c++)
		{
			mat[r*cols+c]=((PRECISIONTYPE)rand()) / ((PRECISIONTYPE)RAND_MAX);;
		}
	}
}

void FUNCNAME(CopyMatrix1D_, PRECISIONTYPE)  (PRECISIONTYPE* srcmat, PRECISIONTYPE* destmat, int rows, int cols)
{
  memcpy(destmat, srcmat, rows*cols*sizeof(PRECISIONTYPE));
}
