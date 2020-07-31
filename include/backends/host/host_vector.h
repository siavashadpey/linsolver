#ifndef HOST_VECTOR_H
#define HOST_VECTOR_H

#include "backends/base_vector.h"
#include "backends/host/host_matrix.h"

template <typename NumType>
class HostVector: public BaseVector<NumType> {
public:
	/**
	 * Constructor.
	 */
	HostVector();

	/**
	 * Destructor.
	 */
	virtual ~HostVector();

	virtual void allocate(int n);
	virtual void clear();
	virtual void copy(const NumType* w);
	virtual void copy(const BaseVector<NumType>& w);

	virtual NumType& operator[](int i);

	virtual NumType norm() const;
	virtual NumType dot(const BaseVector<NumType>& w) const;
	virtual void zeros();
	virtual void scale(NumType alpha);
	virtual void add(NumType alpha, const BaseVector<NumType>& w, NumType beta);
	virtual void add_scale(NumType alpha, const BaseVector<NumType>& w);
	virtual void scale_add(NumType alpha, const BaseVector<NumType>& w);

protected:
	/**
	 * Values of the vector.
	 */
	NumType* vec_;

private:
	// befriending classes
	friend class HostMatrix<NumType>;

};
#endif