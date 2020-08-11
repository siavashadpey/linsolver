#ifndef BASE_SOLVER_H
#define BASE_SOLVER_H

#include "backends/base_vector.h"
#include "backends/base_matrix.h"

#include "backends/host/host_vector.h"
#include "backends/host/host_matrix.h"

#ifdef __CUDACC__
#include "backends/device/device_vector.h"
#include "backends/device/device_matrix.h"
#endif

/**
 * \brief Base class for Linear Solvers.
 * \tparam MatType Matrix type (HostMatrix<NumType> and DeviceMatrix<NumTyper> currently supported).
 * \tparam VecType Vector type (HostVector<NumType> and DeviceVector<NumTyper> currently supported).
 * \tparam NumType Number type (double and float currently supported).
 */
template <class MatType, class VecType, typename NumType>
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
     * Solves for \p soln in \p mat * \p soln = \p rhs.
     * @param[in]  mat  The matrix in the above equation.
     * @param[in]  rhs  The vector in the above equation.
     * @param[out] soln The vector in the above equation.
     */
    virtual void solve(const MatType& mat, const VecType& rhs, VecType* soln) = 0;
};
#endif
