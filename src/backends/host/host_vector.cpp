#include <cmath>
#include <cassert>

#include "backends/host/host_vector.h"


template <typename NumType>
HostVector<NumType>::HostVector()
	: vec_(nullptr)
{
}

template <typename NumType>
HostVector<NumType>::~HostVector()
{
	this->clear();
}

template <typename NumType>
void HostVector<NumType>::allocate(int n)
{
	assert(n > 0);
	this->clear();

	this->size_ = n;
	this->vec_ = new NumType[n];
}

template <typename NumType>
void HostVector<NumType>::clear()
{
	if (this->size_ > 0) {
		delete[] this->vec_;
		this->size_ = 0;
	}
}

template <typename NumType>
void HostVector<NumType>::copy(const NumType* w)
{
	assert(this->size_ > 0);

	for (int i = 0; i < this->size_; i++) {
		this->vec_[i] = w[i];
	}
}

template <typename NumType>
void HostVector<NumType>::copy(const BaseVector<NumType>& w)
{
	assert(this != &w);
	const HostVector<NumType>* w_host = dynamic_cast<const HostVector<NumType>*>(&w);
	assert(w_host != nullptr);

	if (w_host->size_ != this->size_) {
		this->allocate(w_host->size_);
	}

	for (int i = 0; i < this->size_; i++) {
		this->vec_[i] = w_host->vec_[i];
	}
}

template <typename NumType>
NumType& HostVector<NumType>::operator[](int i)
{
	return this->vec_[i];
}

template <typename NumType>
NumType HostVector<NumType>::norm() const
{
	NumType val = static_cast<NumType>(0);

	for (int i = 0; i < this->size_; i++) {
		val += this->vec_[i]*this->vec_[i];
	}

	return sqrt(val);
}

template <typename NumType>
NumType HostVector<NumType>::dot(const BaseVector<NumType>& w) const
{

	const HostVector<NumType>* w_host = dynamic_cast<const HostVector<NumType>*>(&w);
	assert(w_host != nullptr);
	assert(this->size_ == w_host->size_);

	NumType val = static_cast<NumType>(0);

	for (int i = 0; i < this->size_; i++) {
		val += this->vec_[i]*w_host->vec_[i];
	}
	return val;
}

template <typename NumType>
void HostVector<NumType>::zeros() 
{
	const NumType zero = static_cast<NumType>(0);
	for (int i = 0; i < this->size_; i++) {
		this->vec_[i] = zero;
	}
}

template <typename NumType>
void HostVector<NumType>::scale(NumType alpha)
{
	for (int i = 0; i < this->size_; i++) {
		this->vec_[i] *= alpha;
	}
}

template <typename NumType>
void HostVector<NumType>::add(NumType alpha, const BaseVector<NumType>& w, NumType beta)
{
	const NumType one = static_cast<NumType>(1);

	if (alpha == one)
	{
		this->add_scale(beta, w);
		return;
	}
	else if (beta == one)
	{
		this->scale_add(alpha, w);
		return;
	} 
	
	const HostVector<NumType>* w_host = dynamic_cast<const HostVector<NumType>*>(&w);
	assert(w_host != nullptr);
	assert(this->size_ == w_host->size_);

	for (int i = 0; i < this->size_; i++) {
		this->vec_[i] = alpha*this->vec_[i] + beta*w_host->vec_[i];
	}

}

template <typename NumType>
void HostVector<NumType>::add_scale(NumType alpha, const BaseVector<NumType>& w)
{
	const NumType zero = static_cast<NumType>(0);
	if (alpha == zero) {
		return;
	}

	const HostVector<NumType>* w_host = dynamic_cast<const HostVector<NumType>*>(&w);
	assert(w_host != nullptr);
	assert(this->size_ == w_host->size_);

	for (int i = 0; i < this->size_; i++) {
		this->vec_[i] = this->vec_[i] + alpha*w_host->vec_[i];
	}
}

template <typename NumType>
void HostVector<NumType>::scale_add(NumType alpha, const BaseVector<NumType>& w)
{
	const HostVector<NumType>* w_host = dynamic_cast<const HostVector<NumType>*>(&w);
	assert(w_host != nullptr);
	assert(this->size_ == w_host->size_);

	for (int i = 0; i < this->size_; i++) {
		this->vec_[i] = alpha*this->vec_[i] + w_host->vec_[i];
	}
}


// instantiate template classes
template class HostVector<double>;
template class HostVector<float>;