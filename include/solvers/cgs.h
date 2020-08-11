#ifndef CGS_H
#define CGS_H

#include "solvers/base_iterative_solver.h"

/**
 * \brief Implementation of the Conjugate Gradient Squared (CGS) method.
 * \tparam MatType Matrix type (HostMatrix<NumType> and DeviceMatrix<NumTyper> currently supported).
 * \tparam VecType Vector type (HostVector<NumType> and DeviceVector<NumTyper> currently supported).
 * \tparam NumType Number type (double and float currently supported).
 */
template <class MatType, class VecType, typename NumType>
class CGS: public BaseIterativeSolver<MatType, VecType, NumType> {
public:
    /**
     * Default constructor.
     */
    CGS();

    /**
     * Destructor.
     */
    ~CGS();

    void clear();

    void solve(const MatType& mat, const VecType& rhs, VecType* soln);

private:
    /** 
     * Vectors  used in CGS algorithm.
     */
    VecType r_;
    VecType q_;
    VecType p_;
    VecType u_;
    VecType r_hat_;
    VecType u_hat_;
    VecType v_hat_;
    VecType q_hat_;
    VecType* p_hat_;

    /**
     * Prepare the CGS solver. This is called within solve.
     */
    void prepare_solver_(const MatType& mat);
};
#endif
