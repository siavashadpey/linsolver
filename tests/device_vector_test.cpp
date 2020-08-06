#include <stdio.h>
#include <cmath>

#include "gtest/gtest.h"

#include "backends/device/device_vector.h"
#include "backends/host/host_vector.h"

#define tol 1E-13

TEST(DeviceVector, test_1)
{
    DeviceVector<float> v_device = DeviceVector<float>();

    HostVector<float> v_host = HostVector<float>();
    const int n = 5;
    v_host.allocate(n);

    for (int i = 0; i < 5; i++) {
        v_host[i] = (float)i + 10.f;
    }
    for (int i = 0; i < 5; i++) {
        printf("v_host[%d] = %f \n", i, v_host[i]);
    }
    v_device.copy(v_host);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
