#include <stdio.h>
#include <cmath>

#include "gtest/gtest.h"

#include "backends/host/host_vector.h"

#define tol 1E-13

TEST(HostVector, test_1)
{
	printf("hello\n");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}