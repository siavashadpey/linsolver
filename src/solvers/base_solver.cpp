
#include "solvers/base_solver.h"

template <class MatType, class VecType, typename NumType>
BaseSolver<MatType, VecType, NumType>::BaseSolver()
{
}

template <class MatType, class VecType, typename NumType>
BaseSolver<MatType, VecType, NumType>::~BaseSolver()
{
}

// instantiate template classes
template class BaseSolver<HostMatrix<double>, HostVector<double>, double>;
template class BaseSolver<HostMatrix<float> , HostVector<float> , float>;

#ifdef __CUDACC__
template class BaseSolver<DeviceMatrix<double>, DeviceVector<double>, double>;
template class BaseSolver<DeviceMatrix<float> , DeviceVector<float> , float>;
#endif