/*
 *	zero (ZR) general (GE) matrix of doubles (D)
 */
void DGEZR(double** mat, int rows, int cols)
{
    int i,j;

	for (i=0;i<rows;i++)
	{
		for (j=0;j<cols;j++)
		{
			mat[i][j]=0;
		}
	}
}
