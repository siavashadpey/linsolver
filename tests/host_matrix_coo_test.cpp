#include <stdio.h>
#include <cmath>

#include "gtest/gtest.h"

#include "backends/host/host_matrix_coo.h"
#include "backends/host/host_vector.h"
#include "backends/host/host_matrix.h"

#define tol 1E-13

TEST(HostMatrixCOO, test_1)
{
	HostMatrixCOO<double> A = HostMatrixCOO<double>();
	const std::string filename = "mm_test.mtx";
	bool success = A.read_matrix_market(filename);

	EXPECT_TRUE(success);

	HostVector<double> x = HostVector<double>();
	const int n = 4;
	x.allocate(n);
	double x_val[] = {1., 2., 3., 4.};
	x.copy(x_val);

	HostVector<double> rhs = HostVector<double>();
	rhs.allocate(n);

	A.multiply(x, &rhs);

	double rhs_e[] = { 106., 113., 266., 176.};
	for (int i = 0; i < n; i++) {
		EXPECT_NEAR(rhs_e[i], rhs[i], tol);
	}

	HostMatrix<double> B = HostMatrix<double>();
	A.convert_to_CSR(B);
	B.multiply(x, &rhs);
	for (int i = 0; i < n; i++) {
		EXPECT_NEAR(rhs_e[i], rhs[i], tol);
	}
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}