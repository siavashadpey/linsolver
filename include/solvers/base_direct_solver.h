#ifndef BASE_DIRECT_SOLVER_H
#define BASE_DIRECT_SOLVER_H

#include "solvers/base_solver.h"

/**
 * \brief Base class for Direct Linear Solvers.
 * \tparam MatType Matrix type (HostMatrix<NumType> and DeviceMatrix<NumTyper> currently supported).
 * \tparam VecType Vector type (HostVector<NumType> and DeviceVector<NumTyper> currently supported).
 * \tparam NumType Number type (double and float currently supported).
 */
template <class MatType, class VecType, typename NumType>
class BaseDirectSolver: public BaseSolver<MatType, VecType, NumType> {
public:
    /**
     * Default constructor.
     */
    BaseDirectSolver();

    /**
     * Destructor.
     */
    virtual ~BaseDirectSolver();
};

#endif