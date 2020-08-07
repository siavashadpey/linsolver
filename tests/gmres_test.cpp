#include <stdio.h>
#include "gtest/gtest.h"

#include "gtest/gtest.h"

#include "solvers/gmres.h"
#include "backends/host/host_matrix.h"
#include "backends/host/host_vector.h"

#define tol 1E-9

TEST(GMRES, test_1)
{
    HostMatrix<double> A = HostMatrix<double>();
    int n = 4;
    int nnz = 10;
    A.allocate(n, n, nnz);
    double val[] = {3.0, 1.0, -3.0, -1.0, -5.5, 2.3, 4.3, 0.3, 3.2, -3.0};
    int row_idx[] = {0, 3, 5, 7, 10};
    int col_idx[] = {0, 1, 3, 1, 2, 0, 2, 1, 2, 3};
    A.copy_from(val, row_idx, col_idx);
    
    HostVector<double> b = HostVector<double>();
    b.allocate(n);

    double x_e_vals[] = {9, 6.3, 68, -11.4};
    HostVector<double> x_e = HostVector<double>();
    x_e.allocate(n);
    x_e.copy_from(x_e_vals);
    
    A.multiply(x_e, &b); // b = A*x
    

    HostVector<double> x_soln = HostVector<double>();
    x_soln.allocate(n);
    x_soln.zeros();

    GMRES<double> solver = GMRES<double>();
    
    solver.solve(A, b, &x_soln);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x_soln[i], x_e[i], tol);
    }
}

TEST(GMRES, test_2)
{
    HostMatrix<double> A = HostMatrix<double>();
    const std::string filename = "gre__115.mtx";
    bool success = A.read_matrix_market(filename);

    EXPECT_TRUE(success);

    HostVector<double> x_e = HostVector<double>();
    x_e.allocate(A.n());
    x_e.ones();

    HostVector<double> rhs = HostVector<double>();
    rhs.allocate(A.m());
    A.multiply(x_e, &rhs);

    HostVector<double> x_soln = HostVector<double>();
    x_soln.allocate(A.n());
    x_soln.zeros();

    GMRES<double> solver = GMRES<double>();
    
    solver.solve(A, rhs, &x_soln);

    for (int i = 0; i < A.n(); i++) {
        EXPECT_NEAR(x_soln[i], x_e[i], tol);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
