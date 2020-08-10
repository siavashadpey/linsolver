
#include "solvers/base_iterative_solver.h"

template <class MatType, class VecType, typename NumType>
BaseIterativeSolver<MatType, VecType, NumType>::BaseIterativeSolver()
    :   abs_tol_(1E-12),
        rel_tol_(1E-12),
        max_its_(1E3),
        it_counter_(0),
        init_res_norm_(1E10),
        res_norm_(1E12)
{
}

template <class MatType, class VecType, typename NumType>
BaseIterativeSolver<MatType, VecType, NumType>::~BaseIterativeSolver()
{
}

template <class MatType, class VecType, typename NumType>
void BaseIterativeSolver<MatType, VecType, NumType>::set_abs_tolerance(double abs_tol)
{
    abs_tol_ = abs_tol;
}

template <class MatType, class VecType, typename NumType>
void BaseIterativeSolver<MatType, VecType, NumType>::set_rel_tolerance(double rel_tol)
{
    rel_tol_ = rel_tol;
}

template <class MatType, class VecType, typename NumType>
void BaseIterativeSolver<MatType, VecType, NumType>::set_abs_rel_tolerances(double abs_tol, double rel_tol)
{
    abs_tol_ = abs_tol;
    rel_tol_ = rel_tol;
}

template <class MatType, class VecType, typename NumType>
void BaseIterativeSolver<MatType, VecType, NumType>::set_max_iterations(int max_it)
{
    max_its_ = max_it;
}


template <class MatType, class VecType, typename NumType>
bool BaseIterativeSolver<MatType, VecType, NumType>::is_converged_() const {
    return (res_norm_ <= abs_tol_) or
           (res_norm_ <= rel_tol_*init_res_norm_);
}


// instantiate template classes
template class BaseIterativeSolver<HostMatrix<double>, HostVector<double>, double>;
template class BaseIterativeSolver<HostMatrix<float> , HostVector<float> , float>;

#ifdef __CUDACC__
template class BaseIterativeSolver<DeviceMatrix<double>, DeviceVector<double>, double>;
template class BaseIterativeSolver<DeviceMatrix<float> , DeviceVector<float> , float>;
#endif