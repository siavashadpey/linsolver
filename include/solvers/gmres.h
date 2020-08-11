#ifndef GMRES_H
#define GMRES_H

#include "solvers/base_iterative_solver.h"
#include "backends/host/host_vector.h"

/**
 * \brief Implementation of the Generalize Mimimal RESidual (GMRES) method.
 * \tparam MatType Matrix type (HostMatrix<NumType> and DeviceMatrix<NumTyper> currently supported).
 * \tparam VecType Vector type (HostVector<NumType> and DeviceVector<NumTyper> currently supported).
 * \tparam NumType Number type (double and float currently supported).
 */
template <class MatType, class VecType, typename NumType>
class GMRES: public BaseIterativeSolver<MatType, VecType, NumType> {
public:
    /**
     * Default constructor.
     */
    GMRES();

    /**
     * Destructor.
     */
    ~GMRES();

    void clear();

    /**
     * \param[in] K_dim The dimension (or size) of the Krylov space (default is 20).
     */
    void set_krylov_dimension(int K_dim);

    void solve(const MatType& mat, const VecType& rhs, VecType* soln);

private:
    /** 
     * The dimension (or size) of the Krylov space (default is 20).
     */
    int krylov_dim_;

    /** 
     * Vectors and matrices used in GMRES algorithm.
     */
    HostVector<NumType> c_;
    HostVector<NumType> s_;
    HostVector<NumType> g_;
    HostVector<NumType> H_;
    VecType* V_;

    /**
     * Prepare the GMRES solver. This is called within solve.
     */
    void prepare_solver_(const MatType& mat);

    static void rotate_inplace_(NumType c, NumType s, NumType& h, NumType& hp1);
};
#endif
