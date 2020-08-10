
#include "solvers/base_direct_solver.h"

template <class MatType, class VecType, typename NumType>
BaseDirectSolver<MatType, VecType, NumType>::BaseDirectSolver()
{
}

template <class MatType, class VecType, typename NumType>
BaseDirectSolver<MatType, VecType, NumType>::~BaseDirectSolver()
{
}

// instantiate template classes
template class BaseDirectSolver<HostMatrix<double>, HostVector<double>, double>;
template class BaseDirectSolver<HostMatrix<float> , HostVector<float> , float>;

#ifdef __CUDACC__
template class BaseDirectSolver<DeviceMatrix<double>, DeviceVector<double>, double>;
template class BaseDirectSolver<DeviceMatrix<float> , DeviceVector<float> , float>;
#endif