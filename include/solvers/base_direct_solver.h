#ifndef BASE_DIRECT_SOLVER_H
#define BASE_DIRECT_SOLVER_H

#include "solvers/base_solver.h"

template <typename NumType>
class BaseDirectSolver: public BaseSolver<NumType> {
public:
	BaseDirectSolver();

	virtual ~BaseDirectSolver();
};
#endif