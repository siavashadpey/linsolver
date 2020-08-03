#ifndef BASE_SOLVER_H
#define BASE_SOLVER_H

#include "backends/base_vector.h"
#include "backends/base_matrix.h"

/**
 * \brief Base class for Linear Solvers.
 */
template <typename NumType>
class BaseSolver {
public:
	/**
	 * Default constructor.
	 */
	BaseSolver();

	/**
	 * Destructor.
	 */
	virtual ~BaseSolver();

	/**
	 * Solves for soln in mat*soln = rhs.
	 * @param[in]  mat  The matrix in the above equation.
	 * @param[in]  rhs  The vector in the above equation.
	 * @param[out] soln The vector in the above equation.
	 */
	virtual void solve(const BaseMatrix<NumType>& mat, const BaseVector<NumType>& rhs, BaseVector<NumType>* soln) = 0;
};
#endif