#include "backends/base_matrix.h"

template <typename NumType>
BaseMatrix<NumType>::BaseMatrix()
	: m_(0),
	  n_(0),
	  nnz_(0)
{
}

template <typename NumType>
BaseMatrix<NumType>::~BaseMatrix()
{
}

template <typename NumType>
int BaseMatrix<NumType>::m() const
{
	return m_;
}

template <typename NumType>
int BaseMatrix<NumType>::n() const
{
	return n_;
}

template <typename NumType>
int BaseMatrix<NumType>::nnz() const
{
	return nnz_;
}

template <typename NumType>
bool BaseMatrix<NumType>::is_square() const
{
	return n_ == m_;
}

// instantiate template classes
template class BaseMatrix<double>;
template class BaseMatrix<float>;