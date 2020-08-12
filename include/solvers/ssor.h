#ifndef SSOR_H
#define SSOR_H

#include "solvers/base_preconditioner.h"
/**
 * \brief Symmetric Successive Over-Relaxation (SSOR) preconditoner.
 * \tparam MatType Matrix type (HostMatrix<NumType> and DeviceMatrix<NumTyper> currently supported).
 * \tparam VecType Vector type (HostVector<NumType> and DeviceVector<NumTyper> currently supported).
 * \tparam NumType Number type (double and float currently supported).
 * The relaxation parameter :math:`\omega` is set to 1 here.
 */
template <class MatType, class VecType, typename NumType>
class SSOR: public BasePreconditioner<MatType, VecType, NumType> {
public:
    SSOR();
    ~SSOR();

    void prepare_preconditioner(const MatType& mat);
    void apply(VecType* x) const;
    virtual void apply(const VecType& b, VecType* x) const;

protected:
    const MatType* mat_;
    VecType diag_;
};
#endif