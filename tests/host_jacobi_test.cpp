#include <stdio.h>
#include "gtest/gtest.h"

#include "gtest/gtest.h"

#include "solvers/jacobi.h"
#include "backends/host/host_matrix.h"
#include "backends/host/host_vector.h"

#define tol 1E-9

TEST(host_Jacobi, test_1)
{
    HostMatrix<float> A = HostMatrix<float>();
    int n = 4;
    int nnz = 10;
    A.allocate(n, n, nnz);
    float diag_inv_e[] = {1./3., -1./1.3, 1./4.3, 1./0.5};
    float val[] = {3.0, 1.0, -3.0, -1.3, -5.5, 2.3, 4.3, 0.3, 3.2, 0.5};
    int row_idx[] = {0, 3, 5, 7, 10};
    int col_idx[] = {0, 1, 3, 1, 2, 0, 2, 1, 2, 3};
    A.copy_from(val, row_idx, col_idx);
    
    HostVector<float> b = HostVector<float>();
    b.allocate(n);
    float b_val[] = {3., 2.2, 13., -4.45};
    b.copy_from(b_val);

    auto p = Jacobi<HostMatrix<float>, HostVector<float>, float>();
    p.prepare_preconditioner(A);
    p.apply(&b);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(b[i], b_val[i]*diag_inv_e[i], tol);
    }
}

TEST(host_Jacobi, test_2)
{
    HostMatrix<float> A = HostMatrix<float>();
    int n = 4;
    int nnz = 10;
    A.allocate(n, n, nnz);
    float diag_inv_e[] = {1./3., -1./1.3, 1./4.3, 1./0.5};
    float val[] = {3.0, 1.0, -3.0, -1.3, -5.5, 2.3, 4.3, 0.3, 3.2, 0.5};
    int row_idx[] = {0, 3, 5, 7, 10};
    int col_idx[] = {0, 1, 3, 1, 2, 0, 2, 1, 2, 3};
    A.copy_from(val, row_idx, col_idx);
    
    HostVector<float> b = HostVector<float>();
    b.allocate(n);
    float b_val[] = {3., 2.2, 13., -4.45};
    b.copy_from(b_val);

    auto p = Jacobi<HostMatrix<float>, HostVector<float>, float>();
    p.prepare_preconditioner(A);

    HostVector<float> x = HostVector<float>();
    x.allocate(n);
    p.apply(b, &x);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], b_val[i]*diag_inv_e[i], tol);
    }
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
