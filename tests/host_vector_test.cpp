#include <stdio.h>

#include "backends/host/host_vector.h"

int main(int argc, char const *argv[])
{
	(void) argc;
	(void) argv;

	HostVector<double> v = HostVector<double>();
	int n = 5;
	v.allocate(n);
	printf("%d \n", v.n());
	printf("\n");
	for (int i = 0; i < n; i++) {
		v[i] = (double)i;
	}

	for (int i = 0; i < n; i++) {
		printf("%f\n", v[i]);
	}
	printf("\n");

	printf("%f\n", v.norm());
	printf("%f\n", v.dot(v));
	printf("\n");

	v.scale(2);
	for (int i = 0; i < n; i++) {
		printf("%f\n", v[i]);
	}
	printf("\n");

	HostVector<double> w = HostVector<double>();
	w.allocate(n);
	for (int i = 0; i < n; i++) {
		w[i] = (double) (n - i);
	}

	for (int i = 0; i < n; i++) {
		printf("%f\n", w[i]);
	}
	printf("\n");

	v.add(2, w, 3);
	for (int i = 0; i < n; i++) {
		printf("%f\n", v[i]);
	}
	printf("\n");

	w.add(1, v, 2);
	for (int i = 0; i < n; i++) {
		printf("%f\n", w[i]);
	}
	printf("\n");

	w.add(2, v, 1);
	for (int i = 0; i < n; i++) {
		printf("%f\n", w[i]);
	}
	printf("\n");

	w.add(1, v, 0);
	for (int i = 0; i < n; i++) {
		printf("%f\n", w[i]);
	}
	printf("\n");

	double v_data[] = {10., 8., 15.4, 17, 20.};
	v.copy(v_data);
	for (int i = 0; i < n; i++) {
		printf("%f\n", v[i]);
	}
	printf("\n");

	n = 6;
	w.allocate(n);
	for (int i = 0; i < n; i++) {
		w[i] = (double) (n - i);
	}
	v.copy(w);
	printf("%d\n\n", v.n());
	for (int i = 0; i < n; i++) {
		printf("%f\n", v[i]);
	}
	printf("\n");

	v.zeros();
	for (int i = 0; i < n; i++) {
		printf("%f\n", v[i]);
	}
	printf("\n");


	return 0;
}