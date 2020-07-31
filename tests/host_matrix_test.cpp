#include <stdio.h>

#include "backends/host/host_matrix.h"
#include "backends/host/host_vector.h"

int main(int argc, char const *argv[])
{
	(void) argc;
	(void) argv;

	HostMatrix<double> A = HostMatrix<double>();
	int m = 3;
	int n = 4;
	int nnz = 7;

	A.allocate(m, n, nnz);

	printf("%d %d %d \n", A.m(), A.n(), A.nnz());
	printf("%d \n\n", A.is_square());

	double val[] = {2.0, 9.0, -1., 5.5, 6., 7.3, 3.3};
	int row_ptr[] = {0, 2, 5, 7};
	int col_idx[] = {1, 3, 0, 2, 3, 1, 3};
	A.copy(val, row_ptr, col_idx);
	
	printf("%f\n\n", A.norm());

	A.scale(3.0);
	printf("%f\n\n", A.norm());

	A.scale(1.0);
	printf("%f\n\n", A.norm());


	HostMatrix<double> B = HostMatrix<double>();
	n = 4;
	nnz = 11;
	B.allocate(n, n, nnz);
	double B_val[] = {3.0, 2.0, -1., -5.5, 6.2, 2.3, 4.3, 13.2, 0.3, 0, -3.};
	int B_row_idx[] = {0, 2, 5, 8, 11};
	int B_col_idx[] = {0, 2, 1, 2, 3, 0, 2, 3, 1, 2, 3};
	B.copy(B_val, B_row_idx, B_col_idx);

	A.copy(B);

	HostVector<double> x = HostVector<double>();
	HostVector<double> b = HostVector<double>();
	x.allocate(n);
	b.allocate(n);

	for (int i = 0; i < n; i++) {
		x[i] = (double) i+1;
	}

	A.multiply(x,&b);
	for (int i = 0; i < n; i++) {
		printf("%f\n",b[i]);
	}
	printf("\n");


	return 0;
}