/*
 *	zero (ZR) general (GE) matrix of doubles (D)
 */
void DGEZR_1D(double* mat, int rows, int cols)
{
    int i,j;

	for (i=0;i<rows;i++)
	{
		for (j=0;j<cols;j++)
		{
			mat[i*cols+j]=0;
		}
	}
}
