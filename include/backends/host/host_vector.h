#ifndef HOST_VECTOR_H
#define HOST_VECTOR_H

#include "backends/base_vector.h"
#include "backends/host/host_matrix.h"

/**
 * \brief Implementation of a Vector class on the host system.
 */
template <typename NumType>
class HostVector: public BaseVector<NumType> {
public:
	/**
	 * Default constructor.
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

	/**
	 * Return the i-th element of the vector.
	 * @param[in] i The index of the element of interest in the vector.
	 * \return The i-th value of the vector (return by reference).
	 */
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