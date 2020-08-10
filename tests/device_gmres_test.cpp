#include <stdio.h>
#include "gtest/gtest.h"

#include "gtest/gtest.h"

#include "base/cuda_header.cuh"
#include "solvers/gmres.h"
#include "backends/device/device_matrix.h"
#include "backends/device/device_vector.h"
#include "backends/host/host_vector.h"

#define double_tol 1E-5

TEST(device_GMRES, test_1)
{
    HostMatrix<double> A_h = HostMatrix<double>();
    int n = 4;
    int nnz = 10;
    A_h.allocate(n, n, nnz);
    double val[] = {3.0f, 1.0f, -3.0f, -1.0f, -5.5f, 2.3f, 4.3f, 0.3f, 3.2f, -3.0f};
    int row_idx[] = {0, 3, 5, 7, 10};
    int col_idx[] = {0, 1, 3, 1, 2, 0, 2, 1, 2, 3};
    A_h.copy_from(val, row_idx, col_idx);
    
    DeviceMatrix<double> A_d = DeviceMatrix<double>();
    A_d.copy_from(A_h);

    auto b_d = DeviceVector<double>();
    b_d.allocate(n);

    double x_e_vals[] = {9, 6.3, 68, -11.4};
    HostVector<double> x_e_h = HostVector<double>();
    x_e_h.allocate(n);
    x_e_h.copy_from(x_e_vals);

    DeviceVector<double> x_e_d = DeviceVector<double>();
    x_e_d.allocate(n);
    x_e_d.copy_from(x_e_h);
    A_d.multiply(x_e_d, &b_d); // b = A*x
    DeviceVector<double> x_soln_d = DeviceVector<double>();
    x_soln_d.allocate(n);
    x_soln_d.zeros();

    GMRES<DeviceMatrix<double>, DeviceVector<double>, double> solver = GMRES<DeviceMatrix<double>, DeviceVector<double>, double>();
    
    solver.solve(A_d, b_d, &x_soln_d);
    HostVector<double> x_soln_h = HostVector<double>();
    x_soln_d.copy_to(x_soln_h);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x_soln_h[i], x_e_h[i], double_tol);
    }
}

TEST(device_GMRES, test_2)
{
    auto A_h = HostMatrix<double>();
    const std::string filename = "gre__115.mtx";
    bool success = A_h.read_matrix_market(filename);

    EXPECT_TRUE(success);

    auto A_d = DeviceMatrix<double>();
    A_d.copy_from(A_h);

    auto x_e_d = DeviceVector<double>();
    x_e_d.allocate(A_d.n());
    x_e_d.ones();

    auto rhs_d = DeviceVector<double>();
    rhs_d.allocate(A_d.m());
    A_d.multiply(x_e_d, &rhs_d);

    auto x_soln_d = DeviceVector<double>();
    x_soln_d.allocate(A_d.n());
    x_soln_d.zeros();

    auto solver = GMRES<DeviceMatrix<double>, DeviceVector<double>, double>();
    
    solver.solve(A_d, rhs_d, &x_soln_d);

    auto x_soln_h = HostVector<double>();
    x_soln_d.copy_to(x_soln_h);

    auto x_e_h = HostVector<double>();
    x_e_d.copy_to(x_e_h);

    for (int i = 0; i < A_d.n(); i++) {
        EXPECT_NEAR(x_soln_h[i], x_e_h[i], double_tol);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
