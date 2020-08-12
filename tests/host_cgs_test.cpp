#include <stdio.h>
#include "gtest/gtest.h"

#include "gtest/gtest.h"

#include "solvers/cgs.h"
#include "backends/host/host_matrix.h"
#include "backends/host/host_vector.h"
#include "solvers/jacobi.h"

#define tol 2E-8

TEST(host_CGS, test_1)
{
    HostMatrix<double> A = HostMatrix<double>();
    int n = 4;
    int nnz = 8;
    A.allocate(n, n, nnz);
    double val[] = {3.0, 1.5, 2.2, 0.5, 1.5, 4.2, 0.5, 6.02};
    int row_idx[] = {0, 2, 4, 6, 8};
    int col_idx[] = {0, 2, 1, 3, 0, 2, 1, 3};
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

    auto solver = CGS<HostMatrix<double>, HostVector<double>, double>();

    solver.solve(A, b, &x_soln);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x_soln[i], x_e[i], tol);
    }
}

TEST(host_CGS, test_2)
{
    HostMatrix<double> A = HostMatrix<double>();
    int n = 4;
    int nnz = 8;
    A.allocate(n, n, nnz);
    double val[] = {3.0, 1.5, 2.2, 0.5, 1.5, 4.2, 0.5, 6.02};
    int row_idx[] = {0, 2, 4, 6, 8};
    int col_idx[] = {0, 2, 1, 3, 0, 2, 1, 3};
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

    auto solver = CGS<HostMatrix<double>, HostVector<double>, double>();
    auto precond = Jacobi<HostMatrix<double>, HostVector<double>, double>();
    solver.set_preconditioner(precond);
    
    solver.solve(A, b, &x_soln);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x_soln[i], x_e[i], tol);
    }
}

TEST(host_CGS, test_3)
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

    auto solver = CGS<HostMatrix<double>, HostVector<double>, double>();

    solver.solve(A, rhs, &x_soln);

    for (int i = 0; i < A.n(); i++) {
        EXPECT_NEAR(x_soln[i], x_e[i], tol);
    }
}

TEST(host_CGS, test_4)
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

    auto solver =     CGS<HostMatrix<double>, HostVector<double>, double>();
    auto precond = Jacobi<HostMatrix<double>, HostVector<double>, double>();
    solver.set_preconditioner(precond);
    
    solver.solve(A, rhs, &x_soln);

    for (int i = 0; i < A.n(); i++) {
        EXPECT_NEAR(x_soln[i], x_e[i], tol);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}