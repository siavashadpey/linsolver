#ifndef JACOBI_H
#define JACOBI_H

#include "solvers/base_preconditioner.h"
/**
 * \brief Jacobi preconditoner.
 * \tparam MatType Matrix type (HostMatrix<NumType> and DeviceMatrix<NumTyper> currently supported).
 * \tparam VecType Vector type (HostVector<NumType> and DeviceVector<NumTyper> currently supported).
 * \tparam NumType Number type (double and float currently supported).
 */
template <class MatType, class VecType, typename NumType>
class Jacobi: public BasePreconditioner<MatType, VecType, NumType> {
public:
    Jacobi();
    ~Jacobi();

    void prepare_preconditioner(const MatType& mat);

    void apply(VecType* x) const;
    virtual void apply(const VecType& b, VecType* x) const;

protected:
    VecType inv_diag_;
};
#endif