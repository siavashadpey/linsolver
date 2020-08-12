#include <stdio.h>
#include "gtest/gtest.h"

#include "gtest/gtest.h"

#include "base/backend_wrapper.h"
#include "solvers/cgs.h"
#include "backends/device/device_matrix.h"
#include "backends/device/device_vector.h"
#include "backends/host/host_vector.h"
#include "solvers/jacobi.h"

#define tol 2E-8


TEST(device_CGS, test_1)
{
    auto A_h = HostMatrix<float>();
    const std::string filename = "gre__115.mtx";
    bool success = A_h.read_matrix_market(filename);

    EXPECT_TRUE(success);

    auto A_d = DeviceMatrix<float>();
    A_d.copy_from(A_h);

    auto x_e_d = DeviceVector<float>();
    x_e_d.allocate(A_d.n());
    x_e_d.ones();

    auto rhs_d = DeviceVector<float>();
    rhs_d.allocate(A_d.m());
    A_d.multiply(x_e_d, &rhs_d);

    auto x_soln_d = DeviceVector<float>();
    x_soln_d.allocate(A_d.n());
    x_soln_d.zeros();

    auto solver = GMRES<DeviceMatrix<float>, DeviceVector<float>, float>();
    auto precond = Jacobi<HostMatrix<double>, HostVector<double>, double>();
    solver.set_preconditioner(precond);

    solver.solve(A_d, rhs_d, &x_soln_d);

    auto x_soln_h = HostVector<float>();
    x_soln_d.copy_to(x_soln_h);

    auto x_e_h = HostVector<float>();
    x_e_d.copy_to(x_e_h);

    for (int i = 0; i < A_d.n(); i++) {
        EXPECT_NEAR(x_soln_h[i], x_e_h[i], float_tol);
    }

    manager::stop_backend();

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}