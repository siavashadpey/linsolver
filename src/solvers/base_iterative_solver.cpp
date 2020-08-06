
#include "solvers/base_iterative_solver.h"

template <typename NumType>
BaseIterativeSolver<NumType>::BaseIterativeSolver()
    :   abs_tol_(1E-12),
        rel_tol_(1E-12),
        max_its_(1E3),
        it_counter_(0),
        init_res_norm_(1E10),
        res_norm_(1E12)
{
}

template <typename NumType>
BaseIterativeSolver<NumType>::~BaseIterativeSolver()
{
}

template <typename NumType>
void BaseIterativeSolver<NumType>::set_abs_tolerance(double abs_tol)
{
    abs_tol_ = abs_tol;
}

template <typename NumType>
void BaseIterativeSolver<NumType>::set_rel_tolerance(double rel_tol)
{
    rel_tol_ = rel_tol;
}

template <typename NumType>
void BaseIterativeSolver<NumType>::set_abs_rel_tolerances(double abs_tol, double rel_tol)
{
    abs_tol_ = abs_tol;
    rel_tol_ = rel_tol;
}

template <typename NumType>
void BaseIterativeSolver<NumType>::set_max_iterations(int max_it)
{
    max_its_ = max_it;
}


template <typename NumType>
bool BaseIterativeSolver<NumType>::is_converged_() const {
    return (res_norm_ <= abs_tol_) or
           (res_norm_ <= rel_tol_*init_res_norm_);
}


// instantiate template classes
template class BaseIterativeSolver<double>;
template class BaseIterativeSolver<float>;
