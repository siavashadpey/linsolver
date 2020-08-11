
#include "solvers/base_preconditioner.h"

template <class MatType, class VecType, typename NumType>
BasePreconditioner<MatType, VecType, NumType>::BasePreconditioner()
{
}

template <class MatType, class VecType, typename NumType>
BasePreconditioner<MatType, VecType, NumType>::~BasePreconditioner()
{
}

// instantiate template classes
template class BasePreconditioner<HostMatrix<double>, HostVector<double>, double>;
template class BasePreconditioner<HostMatrix<float> , HostVector<float> , float>;

#ifdef __CUDACC__
template class BasePreconditioner<DeviceMatrix<double>, DeviceVector<double>, double>;
template class BasePreconditioner<DeviceMatrix<float> , DeviceVector<float> , float>;
#endif