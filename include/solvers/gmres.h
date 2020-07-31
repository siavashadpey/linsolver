#ifndef GMRES_H
#define GMRES_H

#include "solvers/base_iterative_solver.h"
#include "backends/host/host_vector.h"

template <typename NumType>
class GMRES: public BaseIterativeSolver<NumType> {
public:
	GMRES();
	~GMRES();

	void clear();
	void set_krylov_dimension(int K_dim);

	void solve(const BaseMatrix<NumType>& mat, const BaseVector<NumType>& rhs, BaseVector<NumType>* soln);

protected:
	int krylov_dim_;

	HostVector<NumType> c_;
	HostVector<NumType> s_;
	HostVector<NumType> g_;
	HostVector<NumType> H_;
	HostVector<NumType>* V_;

	void prepare_solver_(int soln_dim);
	void rotate_inplace_(NumType c, NumType s, NumType& h, NumType& hp1) const;
};
#endif