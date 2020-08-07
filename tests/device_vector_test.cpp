#include <stdio.h>
#include <cmath>

#include "gtest/gtest.h"

#include "base/cuda_header.cuh"
#include "backends/device/device_vector.h"
#include "backends/host/host_vector.h"

#define float_tol 1E-5

TEST(DeviceVector, test_1)
{
    DeviceVector<float> v_device = DeviceVector<float>();
    
    HostVector<float> v_host = HostVector<float>();
    
    const int n = 5;
    v_host.allocate(n);
    float norm_e = 0.f;
    for (int i = 0; i < n; i++) {
        v_host[i] = (float)i + 10.f;
        norm_e += v_host[i]*v_host[i];
    }
    norm_e = sqrtf(norm_e);

    v_device.copy_from(v_host);
    EXPECT_NEAR(v_device.norm(), norm_e, float_tol);
    const float c = 1.4f;
    v_device.scale(c);
    EXPECT_NEAR(v_device.norm(), c*norm_e, float_tol);
    v_device.zeros();
    EXPECT_NEAR(v_device.norm(), 0.f, float_tol);
    v_device.ones();
    EXPECT_NEAR(v_device.norm(), sqrtf(n), float_tol);
}

TEST(DeviceVector, test_2)
{
    DeviceVector<float> v_device = DeviceVector<float>();

    const int n = 6;
    v_device.allocate(n);
    float v_data[n];
    for (int i = 0; i < n; i++) {
        v_data[i] = (float)i;
    }
    v_device.copy_from_host(v_data);

    const float c = 4.3f;
    v_device.scale(c);

    HostVector<float> v_host = HostVector<float>();
    v_device.copy_to(v_host);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(v_host[i], c*v_data[i], float_tol);
    }

    v_device.copy_to_host(v_data);
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(v_host[i], v_data[i], float_tol);
    }

}

TEST(DeviceVecotr, test_3)
{
    const int n = 5;

    float* v_data_d;
    CUDA_CALL( cudaMalloc((void**) &v_data_d, n * sizeof(float) ));

    float v_data_h[n];
    for (int i = 0; i < n; i++) {
        v_data_h[i] = (float)i;
    } 
    CUDA_CALL( cudaMemcpy(v_data_d, v_data_h, n * sizeof(float), cudaMemcpyHostToDevice));

    DeviceVector<float> v_device = DeviceVector<float>();
    v_device.allocate(n);
    v_device.copy_from(v_data_d);

    const float c = 1.1;
    v_device.scale(c);

    DeviceVector<float> w_device = DeviceVector<float>();
    v_device.copy_to(w_device);

    float* w_data_d;
    CUDA_CALL( cudaMalloc((void**) &w_data_d, n * sizeof(float) ));
    w_device.copy_to(w_data_d);

    float w_data_h[n];
    CUDA_CALL( cudaMemcpy(w_data_h, w_data_d, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(c*v_data_h[i], w_data_h[i], float_tol);
    } 

    v_device.ones();
    w_device.copy_from(v_device);
    w_device.copy_to_host(v_data_h);
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(v_data_h[i], 1.f, float_tol);
    } 

    CUDA_CALL( cudaFree(v_data_d));
    CUDA_CALL( cudaFree(w_data_d));
}

TEST(DeviceVector, test_4)
{
    HostVector<float> v_h = HostVector<float>();
    HostVector<float> w_h = HostVector<float>();

    const int n = 10;
    v_h.allocate(n);
    w_h.allocate(n);

    for (int i = 0; i < n; i++) {
        v_h[i] = (float)i + 10.f;
        w_h[i] = -(float)i + 3.4f;
    }
    
    DeviceVector<float> v_d = DeviceVector<float>();
    DeviceVector<float> w_d = DeviceVector<float>();    
    
    v_d.copy_from(v_h);
    w_d.copy_from(w_h);

    float res_d = v_d.dot(w_d);
    float res_h = v_h.dot(w_h);
    EXPECT_NEAR(res_d, res_h, float_tol);

    v_d.add(2, w_d, 3);
    float v_data[n];
    v_d.copy_to_host(v_data);

    v_h.add(2, w_h, 3);
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(v_h[i], v_data[i], float_tol);
    }
    
    v_d.add(1.f, w_d, 2.f);
    v_d.copy_to_host(v_data);

    v_h.add(1.f, w_h, 2.f);
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(v_h[i], v_data[i], float_tol);
    }

    v_d.add(3.f, w_d, 1.f);
    v_d.copy_to_host(v_data);

    v_h.add(3.f, w_h, 1.f);
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(v_h[i], v_data[i], float_tol);
    }

    v_d.add(1.f, w_d, 0.f);
    v_d.copy_to_host(v_data);

    v_h.add(1.f, w_h, 0.f);
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(v_h[i], v_data[i], float_tol);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
