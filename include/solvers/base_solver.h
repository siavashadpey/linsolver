#ifndef BASE_SOLVER_H
#define BASE_SOLVER_H

#include "backends/base_vector.h"
#include "backends/base_matrix.h"

template <typename NumType>
class BaseSolver {
public:
	/**
	 * Constructor.
	 */
	BaseSolver();

	/**
	 * Destructor.
	 */
	virtual ~BaseSolver();

	/**
	 * Solves for x in mat*x = rhs.
	 */
	virtual void solve(const BaseMatrix<NumType>& mat, const BaseVector<NumType>& rhs, BaseVector<NumType>* soln) = 0;
};
#endif