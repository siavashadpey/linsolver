#ifndef BASE_PRECONDITIONER_H
#define BASE_PRECONDITIONER_H

#include "solvers/base_solver.h"

/**
 * \brief Base class for Preconditioners.
 * \tparam MatType Matrix type (HostMatrix<NumType> and DeviceMatrix<NumTyper> currently supported).
 * \tparam VecType Vector type (HostVector<NumType> and DeviceVector<NumTyper> currently supported).
 * \tparam NumType Number type (double and float currently supported).
 */
template <class MatType, class VecType, typename NumType>
class BasePreconditioner {
public:
    BasePreconditioner();

    virtual ~BasePreconditioner();

    /**
     * Prepares the preconditioner matrix \p M of matrix \p mat.
     * \param[in] mat Matrix for which the preconditioner is to be used.
     */
    virtual void prepare_preconditioner(const MatType &mat) = 0;

    /**
     * Apply the preconditioner on the provided vector in place. 
     */
    virtual void apply(VecType* x) const = 0;

};
#endif