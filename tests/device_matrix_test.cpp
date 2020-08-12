#include <stdio.h>
#include <cmath>

#include "gtest/gtest.h"

#include "base/cuda_header.cuh"
#include "base/backend_wrapper.h"
#include "backends/device/device_matrix.h"
#include "backends/device/device_vector.h"
#include "backends/host/host_matrix.h"
#include "backends/host/host_vector.h"

#define tol 1E-13
#define float_tol 1E-5

TEST(DeviceMatrix, test_1)
{
    manager::start_backend();

    int m = 3;
    int n = 4;
    int nnz = 7;

    double val[] = {2.0, 9.0, -1., 5.5, 6., 7.3, 3.3};
    int row_ptr[] = {0, 2, 5, 7};
    int col_idx[] = {1, 3, 0, 2, 3, 1, 3};
    HostMatrix<double> A_h = HostMatrix<double>();
    A_h.allocate(m, n, nnz);
    A_h.copy_from(val, row_ptr, col_idx);

    DeviceMatrix<double> A_d = DeviceMatrix<double>();
    A_d.copy_from(A_h);
    
    EXPECT_EQ(A_d.m(), m);
    EXPECT_EQ(A_d.n(), n);
    EXPECT_EQ(A_d.nnz(), nnz);
    EXPECT_FALSE(A_d.is_square());

    double norm_e = 0.;
    for (int i = 0; i < nnz; i++) {
        norm_e += val[i]*val[i];
    }
    norm_e = sqrt(norm_e);
    EXPECT_NEAR(A_d.norm(), norm_e, tol); // frobenius norm

    A_d.scale(3.0);
    norm_e *= 3.0;
    EXPECT_NEAR(A_d.norm(), norm_e, tol); // scale

    A_d.scale(1.0);
    EXPECT_NEAR(A_d.norm(), norm_e, tol); // scale (special scenario)

    manager::stop_backend();
}

TEST(DeviceMatrix, test_2)
{
    manager::start_backend();

    int n = 4;
    int m = n;
    int nnz = 11;

    double val[] =  {3.0, 2.0, -1., -5.5, 6.2, 2.3, 4.3, 13.2, 0.3, 0, -3.};
    int row_ptr[] = {0, 2, 5, 8, 11};
    int col_idx[] = {0, 2, 1, 2, 3, 0, 2, 3, 1, 2, 3};

    double* val_d;
    int* row_ptr_d;
    int* col_idx_d;

    CUDA_CALL( cudaMalloc( (void**) &val_d, nnz * sizeof(double)));
    CUDA_CALL( cudaMalloc( (void**) &row_ptr_d, (m+1) * sizeof(int)));
    CUDA_CALL( cudaMalloc( (void**) &col_idx_d, nnz * sizeof(int)));

    CUDA_CALL( cudaMemcpy(val_d, val, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL( cudaMemcpy(row_ptr_d, row_ptr, (m+1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL( cudaMemcpy(col_idx_d, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));

    DeviceMatrix<double> A_d = DeviceMatrix<double>();
    A_d.allocate(m, n, nnz);
    EXPECT_TRUE(A_d.is_square());

    A_d.copy_from(val_d, row_ptr_d, col_idx_d);

    DeviceMatrix<double> B_d = DeviceMatrix<double>();
    B_d.copy_from(A_d);

    HostVector<double> x_h = HostVector<double>();
    x_h.allocate(n);
    for (int i = 0; i < n; i++) {
        x_h[i] = (double) i+1;
    }

    DeviceVector<double> x_d = DeviceVector<double>();
    x_d.copy_from(x_h);

    DeviceVector<double> b_d = DeviceVector<double>();
    b_d.allocate(n);

    B_d.multiply(x_d, &b_d);

    HostVector<double> b_h = HostVector<double>();
    b_d.copy_to(b_h);

    double b_e[] = {9., 6.3, 68., -11.4};
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(b_h[i], b_e[i], tol); // b = A*x
    }

    B_d.clear();
    EXPECT_EQ(B_d.m(), 0);
    EXPECT_EQ(B_d.n(), 0);
    EXPECT_EQ(B_d.nnz(), 0);    

    CUDA_CALL( cudaFree(val_d));
    CUDA_CALL( cudaFree(row_ptr_d));
    CUDA_CALL( cudaFree(col_idx_d));

    manager::stop_backend();
}

TEST(DeviceMatrix, test_3)
{
    manager::start_backend();

    HostMatrix<double> A_h = HostMatrix<double>();

    int m = 4;
    int n = 4; 
    int nnz = 9;
    A_h.allocate(m, n, nnz);

    double val[] = {-1., 2., -3.2,  4., 7., 10., .4, 3., 1.1};
    int row_ptr[] = {0, 3, 5, 7, 9};
    int col_idx[] = {0, 2, 3, 1, 3, 0, 2, 2, 3};
    double inv_diag_e [] = {-1., 1./4., 1./.4, 1./1.1};
    
    A_h.copy_from(val, row_ptr, col_idx);
    
    DeviceMatrix<double> A_d = DeviceMatrix<double>();
    A_d.copy_from(A_h);

    DeviceVector<double> inv_diag_d = DeviceVector<double>();
    A_d.compute_inverse_diagonals(&inv_diag_d);

    HostVector<double> inv_diag_h = HostVector<double>();
    inv_diag_d.copy_to(inv_diag_h);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(inv_diag_h[i], inv_diag_e[i], tol);
    }

    manager::stop_backend();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
