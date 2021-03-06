#include <cassert>
#include "solvers/jacobi.h"

template <class MatType, class VecType, typename NumType>
Jacobi<MatType, VecType, NumType>::Jacobi()
{
}

template <class MatType, class VecType, typename NumType>
Jacobi<MatType, VecType, NumType>::~Jacobi()
{
}

template <class MatType, class VecType, typename NumType>
void Jacobi<MatType, VecType, NumType>::prepare_preconditioner(const MatType& mat)
{
    assert(mat.is_square());
    mat.compute_inverse_diagonals(&(this->inv_diag_));
}

template <class MatType, class VecType, typename NumType>
void Jacobi<MatType, VecType, NumType>::apply(VecType* x) const
{
    x->elementwise_multiply(this->inv_diag_);
}

template <class MatType, class VecType, typename NumType>
void Jacobi<MatType, VecType, NumType>::apply(const VecType& b, VecType* x) const
{
    x->elementwise_multiply(this->inv_diag_, b);
}

// instantiate template classes
template class Jacobi<HostMatrix<double>, HostVector<double>, double>;
template class Jacobi<HostMatrix<float> , HostVector<float> , float>;

#ifdef __CUDACC__
template class Jacobi<DeviceMatrix<double>, DeviceVector<double>, double>;
template class Jacobi<DeviceMatrix<float> , DeviceVector<float> , float>;
#endif