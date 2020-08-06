#include "backends/base_vector.h"

template <typename NumType>
BaseVector<NumType>::BaseVector()
    :   size_(0)
{
}

template <typename NumType>
BaseVector<NumType>::~BaseVector()
{
}

template <typename NumType>
int BaseVector<NumType>::n() const 
{
    return this->size_;
}

// instantiate template classes
template class BaseVector<double>;
template class BaseVector<float>;
