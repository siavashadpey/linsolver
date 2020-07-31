#include <stdio.h>

#include "solvers/gmres.h"
#include "backends/host/host_matrix.h"
#include "backends/host/host_vector.h"


int main(int argc, char const *argv[])
{
	(void) argc;
	(void) argv;

	int n = 4;

	HostMatrix<double> A = HostMatrix<double>();
	int nnz = 10;
	A.allocate(n, n, nnz);
	double val[] = {3.0, 1.0, -3.0, -1.0, -5.5, 2.3, 4.3, 0.3, 3.2, -3.0};
	int row_idx[] = {0, 3, 5, 7, 10};
	int col_idx[] = {0, 1, 3, 1, 2, 0, 2, 1, 2, 3};
	A.copy(val, row_idx, col_idx);
	
	HostVector<double> b = HostVector<double>();
	b.allocate(n);
	double b_vals[] = {9, 6.3, 68, -11.4};
	b.copy(b_vals);

	HostVector<double> x = HostVector<double>();
	x.allocate(n);
	x.zeros();

	GMRES<double> solver = GMRES<double>();
	solver.set_krylov_dimension(100);
	solver.set_max_iterations(100);
	
	solver.solve(A, b, &x);

	for (int i = 0; i < n; i++) {
		printf("%f\n", x[i]);
	}
	printf("\n");

	HostVector<double> b_gmres = HostVector<double>();
	b_gmres.allocate(n);
	A.multiply(x, &b_gmres);
	for (int i = 0; i < n; i++) {
		printf("%f\n", b_gmres[i]);
	}
	printf("\n");
	
	return 0;
}