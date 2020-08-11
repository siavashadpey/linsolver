#ifndef CG_H
#define CG_H

#include "solvers/base_iterative_solver.h"
#include "backends/host/host_vector.h"

/**
 * \brief Implementation of the Conjugate Gradient (CG) method.
 * \tparam MatType Matrix type (HostMatrix<NumType> and DeviceMatrix<NumTyper> currently supported).
 * \tparam VecType Vector type (HostVector<NumType> and DeviceVector<NumTyper> currently supported).
 * \tparam NumType Number type (double and float currently supported).
 */
template <class MatType, class VecType, typename NumType>
class CG: public BaseIterativeSolver<MatType, VecType, NumType> {
public:
    /**
     * Default constructor.
     */
    CG();

    /**
     * Destructor.
     */
    ~CG();

    /**
     * Clear all dynamically allocated memory by this class' methods.
     */
    void clear();

    void solve(const MatType& mat, const VecType& rhs, VecType* soln);

private:
    /** 
     * Vectors  used in CG algorithm.
     */
    VecType r_;
    VecType p_;
    VecType q_;
    VecType* z_;
    

    /**
     * Prepare the CG solver. This is called within solve.
     */
    void prepare_solver_(int soln_dim);
};
#endif
