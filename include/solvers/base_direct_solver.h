#ifndef BASE_DIRECT_SOLVER_H
#define BASE_DIRECT_SOLVER_H

#include "solvers/base_solver.h"

/**
 * \brief Base class for Direct Linear Solvers.
 */
template <typename NumType>
class BaseDirectSolver: public BaseSolver<NumType> {
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