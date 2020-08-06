#ifndef BASE_ITERATIVE_SOLVER_H
#define BASE_ITERATIVE_SOLVER_H

#include "solvers/base_solver.h"

/**
 * \brief Base class for Iterative Linear Solvers.
 * 
 * Manages tasks and information related to convergence.
 */
template <typename NumType>
class BaseIterativeSolver: public BaseSolver<NumType> {
public:
    BaseIterativeSolver();

    virtual ~BaseIterativeSolver();

    /** 
     * @param[in] abs_tol The absolute tolerance threshold for convergence (default is 1E-12).
     * \see is_converged_.
     */
    void set_abs_tolerance(double abs_tol);

    /** 
     * @param[in] rel_tol The relative tolerance threshold for convergence (default is 1E-12).
     * \see is_converged_.
     */
    void set_rel_tolerance(double rel_tol);

    /** 
     * @param[in] abs_tol The absolute tolerance threshold for convergence (default is 1E-12).
     * @param[in] rel_tol The relative tolerance threshold for convergence (default is 1E-12).
     * \see is_converged_.
     */
    void set_abs_rel_tolerances(double abs_tol, double rel_tol);

    /**
     * @param[in] max_it The maximum number of iterations before 
     * stopping, if the solution has not yet converged (default is 1000).
     */
    void set_max_iterations(int max_it);

protected:
    /**
     * Absolute tolerance threshold (default is 1E-12).
     */
    double abs_tol_;

    /**
     * Relative tolerance threshold (default is 1E-12).
     */
    double rel_tol_;

    /**
     * Maximum number of iterations allowed (default is 1000).
     */
    int max_its_;

    /**
     * Keeps count of the current iteration.
     */
    int it_counter_;

    /**
     * L2 norm of the initial residual.
     */
    double init_res_norm_;

    /**
     * L2 residual norm of the current iteration.
     */
    double res_norm_;


    /**
     * \return If the solution has converged or not.
     *
     * The solution has converged if one of the two following conditions is true:
     *     1. The L2 norm of the residual is smaller than the absolute tolerance.
     *     2. The ratio of the L2 norm of the residual to the L2 norm
     *      of the initial residual is smaller than the relative tolerance.
     */
    bool is_converged_() const;
};

#endif
