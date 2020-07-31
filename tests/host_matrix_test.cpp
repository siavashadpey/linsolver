#include <stdio.h>
#include <cmath>

#include "gtest/gtest.h"

#include "backends/host/host_matrix.h"
#include "backends/host/host_vector.h"

#define tol 1E-13

TEST(HostMatrix, test_1)
{

	HostMatrix<double> A = HostMatrix<double>();
	int m = 3;
	int n = 4;
	int nnz = 7;

	A.allocate(m, n, nnz);
	EXPECT_EQ(A.m(), m);
	EXPECT_EQ(A.n(), n);
	EXPECT_EQ(A.nnz(), nnz);
	EXPECT_FALSE(A.is_square());

	double val[] = {2.0, 9.0, -1., 5.5, 6., 7.3, 3.3};
	int row_ptr[] = {0, 2, 5, 7};
	int col_idx[] = {1, 3, 0, 2, 3, 1, 3};
	A.copy(val, row_ptr, col_idx);

	double norm_e = 0.;
	for (int i = 0; i < nnz; i++) {
		norm_e += val[i]*val[i];
	}
	norm_e = sqrt(norm_e);
	EXPECT_NEAR(A.norm(), norm_e, tol); // frobenius norm

	A.scale(3.0);
	norm_e *= 3.0;
	EXPECT_NEAR(A.norm(), norm_e, tol); // scale

	A.scale(1.0);
	EXPECT_NEAR(A.norm(), norm_e, tol); // scale (special scenario)

	HostMatrix<double> B = HostMatrix<double>();
	n = 4;
	nnz = 11;
	B.allocate(n, n, nnz);
	double B_val[] = {3.0, 2.0, -1., -5.5, 6.2, 2.3, 4.3, 13.2, 0.3, 0, -3.};
	int B_row_idx[] = {0, 2, 5, 8, 11};
	int B_col_idx[] = {0, 2, 1, 2, 3, 0, 2, 3, 1, 2, 3};
	B.copy(B_val, B_row_idx, B_col_idx);

	A.copy(B); 

	// copy from another class instance
	EXPECT_EQ(A.m(), n);
	EXPECT_EQ(A.n(), n);
	EXPECT_EQ(A.nnz(), nnz);
	EXPECT_TRUE(A.is_square());

	HostVector<double> x = HostVector<double>();
	HostVector<double> b = HostVector<double>();
	x.allocate(n);
	b.allocate(n);

	for (int i = 0; i < n; i++) {
		x[i] = (double) i+1;
	}

	A.multiply(x,&b);

	double b_e[] = {9., 6.3, 68., -11.4};
	for (int i = 0; i < n; i++) {
		EXPECT_NEAR(b[i], b_e[i], tol); // b = A*x
	}

	A.clear();
	EXPECT_EQ(A.m(), 0);
	EXPECT_EQ(A.n(), 0);
	EXPECT_EQ(A.nnz(), 0);

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}