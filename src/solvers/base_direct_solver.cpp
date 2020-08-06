
#include "solvers/base_direct_solver.h"
#include "backends/host/host_vector.h"
#include "backends/host/host_matrix.h"

template <typename NumType>
BaseDirectSolver<NumType>::BaseDirectSolver()
{
}

template <typename NumType>
BaseDirectSolver<NumType>::~BaseDirectSolver()
{
}

// instantiate template classes
template class BaseDirectSolver<double>;
template class BaseDirectSolver<float>; 
