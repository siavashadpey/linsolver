
#include "solvers/base_solver.h"
#include "backends/host/host_vector.h"
#include "backends/host/host_matrix.h"

template <typename NumType>
BaseSolver<NumType>::BaseSolver()
{
}

template <typename NumType>
BaseSolver<NumType>::~BaseSolver()
{
}

// instantiate template classes
template class BaseSolver<double>;
template class BaseSolver<float>; 