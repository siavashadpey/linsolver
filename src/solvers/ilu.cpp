#include <cassert>

#include "base/error.h"
#include "solvers/ilu.h"
#include "base/error.h"

template <class MatType, class VecType, typename NumType>
ILU<MatType, VecType, NumType>::ILU()
    :   p_(0),
        tau_(1E-10)
{
}

template <class MatType, class VecType, typename NumType>
ILU<MatType, VecType, NumType>::~ILU()
{
}

template <class MatType, class VecType, typename NumType>
void ILU<MatType, VecType, NumType>::set_level(int level)
{
    Error("ILUT is not currently implemented. Only ILU0.");
    p_ = level;
}

template <class MatType, class VecType, typename NumType>
void ILU<MatType, VecType, NumType>::set_threshold(int tau)
{
    Error("ILUT is not currently implemented. Only ILU0.");
    tau_ = tau;
}

template <class MatType, class VecType, typename NumType>
void ILU<MatType, VecType, NumType>::prepare_preconditioner(const MatType& mat)
{
    ilu_mat_.copy_from(mat);
    assert(ilu_mat_.is_square());
    
    ilu_mat_.ILU0_factorize();
}

template <class MatType, class VecType, typename NumType>
void ILU<MatType, VecType, NumType>::apply(VecType* x) const
{
    VecType b;
    b.copy_from(*x);
    apply(b, x);
}

template <class MatType, class VecType, typename NumType>
void ILU<MatType, VecType, NumType>::apply(const VecType& b, VecType* x) const
{
    ilu_mat_.lower_upper_solve(b, x);
}

// instantiate template classes
template class ILU<HostMatrix<double>, HostVector<double>, double>;
template class ILU<HostMatrix<float> , HostVector<float> , float>;

#ifdef __CUDACC__
template class ILU<DeviceMatrix<double>, DeviceVector<double>, double>;
template class ILU<DeviceMatrix<float> , DeviceVector<float> , float>;
#endif