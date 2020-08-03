#include <stdio.h>
#include <cmath>

#include "gtest/gtest.h"

#include "backends/host/host_vector.h"

#define tol 1E-13

TEST(HostVector, test_1)
{
	HostVector<double> v = HostVector<double>();
	const int n = 5;
	v.allocate(n);
	double v_e[n];
	double norm_e = 0.0;
	for (int i = 0; i < n; i++) {
		v[i] = (double)i;
		v_e[i] = (double)i;
		norm_e += v_e[i]*v_e[i];
	}
	norm_e = sqrt(norm_e);

	EXPECT_EQ(n+1, v.n()); // size
	EXPECT_NEAR(v.norm(), norm_e, tol); // l2-norm computation
	for (int i = 0; i < n; i++) {
		EXPECT_NEAR(v[i], v_e[i], tol); // values
	}
	
	const double c = 2.4;
	v.scale(c);
	for (int i = 0; i < n; i++) {
		v_e[i] *= c;
		EXPECT_NEAR(v[i], v_e[i], tol); // scale
	}
	printf("\n");

	HostVector<double> w = HostVector<double>();
	w.allocate(n);
	double w_e[n];
	double v_dot_w = 0.;
	for (int i = 0; i < n; i++) {
		w[i] = (double) 3*(n - i);
		w_e[i] = (double) 3*(n - i);
		v_dot_w += w_e[i]*v_e[i];
	}
	EXPECT_NEAR(v.dot(w), v_dot_w, tol); // dot product


	v.add(2, w, 3);
	for (int i = 0; i < n; i++) {
		v_e[i] = 2.*v_e[i] + 3.*w_e[i];
		EXPECT_NEAR(v[i], v_e[i], tol); // v = a*v + b*w
	}

	w.add(1, v, 2);
	for (int i = 0; i < n; i++) {
		w_e[i] += 2.*v_e[i];
		EXPECT_NEAR(w[i], w_e[i], tol); // v = v + b*w
	}

	w.add(2, v, 1);
	for (int i = 0; i < n; i++) {
		w_e[i] = 2.*w_e[i] + v_e[i];
		EXPECT_NEAR(w[i], w_e[i], tol); // v = a*v + w
	}

	w.add(1, v, 0);
	for (int i = 0; i < n; i++) {
		EXPECT_NEAR(w[i], w_e[i], tol); // v = v
	}

	double v_data[] = {10., 8., 15.4, 17, 20.};
	v.copy(v_data);
	for (int i = 0; i < n; i++) {
		EXPECT_NEAR(v[i], v_data[i], tol); // copy from raw data
	}

	const int n_new = 6;
	w.allocate(n_new);
	double w_new[n_new];
	for (int i = 0; i < n_new; i++) {
		w[i] = (double) 4.*(n_new - i);
		w_new[i] = (double) 4.*(n_new - i);
	}

	v.copy(w);
	EXPECT_EQ(v.n(), n_new); // size
	for (int i = 0; i < n; i++) {
		EXPECT_NEAR(v[i], w_new[i], tol); // copy from another class instance
	}

	v.zeros();
	for (int i = 0; i < n; i++) {
		EXPECT_NEAR(v[i], 0., tol); // zeroed
	}

	v.clear();
	EXPECT_EQ(v.n(), 0); // cleared data

	double* a = new double[4];
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}