#ifndef BASE_ITERATIVE_SOLVER_H
#define BASE_ITERATIVE_SOLVER_H

#include "solvers/base_solver.h"

template <typename NumType>
class BaseIterativeSolver: public BaseSolver<NumType> {
public:
	BaseIterativeSolver();

	virtual ~BaseIterativeSolver();

	void set_abs_tolerance(double abs_tol);

	void set_rel_tolerance(double rel_tol);

	void set_abs_rel_tolerances(double abs_tol, double rel_tol);

	void set_max_iterations(int max_it);

protected:
	double abs_tol_;
	double rel_tol_;
	int max_its_;
	int it_counter_;
	double init_res_norm_;
	double res_norm_;

	NumType norm_(const BaseVector<NumType>& v) const;
	bool is_converged_() const;
};

#endif