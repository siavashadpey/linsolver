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
    A.copy_from(val, row_ptr, col_idx);

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
    B.copy_from(B_val, B_row_idx, B_col_idx);

    A.copy_from(B); 

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

TEST(HostMatrix, test_2) 
{
    HostMatrix<double> A = HostMatrix<double>();
    const std::string filename = "mm_test.mtx";
    bool success = A.read_matrix_market(filename);

    EXPECT_TRUE(success);

    HostVector<double> x = HostVector<double>();
    const int n = 4;
    x.allocate(n);
    double x_val[] = {1., 2., 3., 4.};
    x.copy_from(x_val);

    HostVector<double> rhs = HostVector<double>();
    rhs.allocate(n);

    A.multiply(x, &rhs);

    double rhs_e[] = { 106., 113., 266., 176.};
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(rhs_e[i], rhs[i], tol);
    }
}

TEST(HostMatrix, test_3)
{
    HostMatrix<double> A = HostMatrix<double>();

    int m = 4;
    int n = 4; 
    int nnz = 9;
    A.allocate(m, n, nnz);

    double val[] = {-1., 2., -3.2,  4., 7., 10., .4, 3., 1.1};
    int row_ptr[] = {0, 3, 5, 7, 9};
    int col_idx[] = {0, 2, 3, 1, 3, 0, 2, 2, 3};
    double diag_e [] = {-1., 4., .4, 1.1};
    
    A.copy_from(val, row_ptr, col_idx);

    HostVector<double> diag = HostVector<double>();
    A.get_diagonals(&diag);

    HostVector<double> inv_diag = HostVector<double>();
    A.compute_inverse_diagonals(&inv_diag);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(diag[i], diag_e[i], tol);
        EXPECT_NEAR(inv_diag[i], 1./diag_e[i], tol);
    }
}

TEST(HostMatrix, test_4)
{
    HostMatrix<double> A = HostMatrix<double>();
    int n = 4;
    int nnz = 7;
    A.allocate(n, n, nnz);
    double val[] = {3., -2.1, 1.5, 0.5, 4.5, 4.3, 6.05};
    int row_idx[] = {0, 1, 3, 5, 7};
    int col_idx[] = {0, 1, 0, 0, 2, 1, 3};
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

    A.lower_solve(b, &x_soln);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x_soln[i], x_e[i], tol);
    }
}

TEST(HostMatrix, test_5)
{
    HostMatrix<double> A = HostMatrix<double>();
    int n = 4;
    int nnz = 8;
    A.allocate(n, n, nnz);
    double val[] = {3., 0.5, 1.5, 4.5, -2.1, 4.75, 4.3, 6.05};
    int row_idx[] = {0, 3, 5, 7, 8};
    int col_idx[] = {0, 2, 1, 3, 1, 2, 3, 3};
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

    A.upper_solve(b, &x_soln);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x_soln[i], x_e[i], tol);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
