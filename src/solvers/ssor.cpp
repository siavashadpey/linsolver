#include <cassert>

#include "solvers/ssor.h"
#include "base/error.h"

template <class MatType, class VecType, typename NumType>
SSOR<MatType, VecType, NumType>::SSOR()
{
}

template <class MatType, class VecType, typename NumType>
SSOR<MatType, VecType, NumType>::~SSOR()
{
}

template <class MatType, class VecType, typename NumType>
void SSOR<MatType, VecType, NumType>::prepare_preconditioner(const MatType& mat)
{
    mat_ = &mat;
    assert(mat_->is_square());
    mat_->get_diagonals(&(this->diag_));
}

template <class MatType, class VecType, typename NumType>
void SSOR<MatType, VecType, NumType>::apply(VecType* x) const
{
    VecType b;
    b.copy_from(*x);
    apply(b, x);
}

template <class MatType, class VecType, typename NumType>
void SSOR<MatType, VecType, NumType>::apply(const VecType& b, VecType* x) const
{
    // We solve for x in (D + L) * D^-1 * (D + U) * x = b

    // Solve for y in (D + L) * y = b 
    // for memory efficiency, use x as place of y
    mat_->lower_solve(b, x);

    // Solve for z in D^-1 * z = y (in place)
    x->elementwise_multiply(this->diag_);

    // Finally, sol for x in (D + U) * x = z
    mat_->upper_solve(*x, x);
}

// instantiate template classes
template class SSOR<HostMatrix<double>, HostVector<double>, double>;
template class SSOR<HostMatrix<float> , HostVector<float> , float>;

#ifdef __CUDACC__
template class SSOR<DeviceMatrix<double>, DeviceVector<double>, double>;
template class SSOR<DeviceMatrix<float> , DeviceVector<float> , float>;
#endif