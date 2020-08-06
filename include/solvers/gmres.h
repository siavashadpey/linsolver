#ifndef GMRES_H
#define GMRES_H

#include "solvers/base_iterative_solver.h"
#include "backends/host/host_vector.h"

/**
 * \brief Implementation of the Generalize Mimimal RESidual (GMRES) method.
 */
template <typename NumType>
class GMRES: public BaseIterativeSolver<NumType> {
public:
    /**
     * Default constructor.
     */
    GMRES();

    /**
     * Destructor.
     */
    ~GMRES();

    /**
     * Clear all dynamically allocated memory by this class' methods.
     */
    void clear();

    /**
     * \param[in] K_dim The dimension (or size) of the Krylov space (default is 20).
     */
    void set_krylov_dimension(int K_dim);

    void solve(const BaseMatrix<NumType>& mat, const BaseVector<NumType>& rhs, BaseVector<NumType>* soln);

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
    HostVector<NumType>* V_;

    /**
     * Prepare the GMRES solver. This is called within solve.
     */
    void prepare_solver_(int soln_dim);

    void rotate_inplace_(NumType c, NumType s, NumType& h, NumType& hp1) const;
};
#endif
