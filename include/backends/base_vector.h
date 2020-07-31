#ifndef BASE_VECTOR_H
#define BASE_VECTOR_H

template <typename NumType>
class BaseVector {
public:
	/**
	 * Constructor.
	 */
	BaseVector();

	/**
	 * Destructor.
	 */
	virtual ~BaseVector();

	/**
	 * Allocate a size of n to the vector.
	 */
	virtual void allocate(int n) = 0;

	/**
	 * Clear vector.
	 */
	virtual void clear() = 0;

	/**
	 * Copy data to the vector.
	 * allocate() should be called first.
	 */
	virtual void copy(const NumType* w) = 0;

	/**
	 * v = w, where v is this.
	 */
	virtual void copy(const BaseVector<NumType>& w) = 0;

	/** 
	 * Returns the size of the vector.
	 */
	int n() const;

	/**
	 * Returns the l2 norm of the vector.
	 */
	virtual NumType norm() const = 0;

	/**
	 * Returns the dot product between v and w, where v is this.
	 */
	virtual NumType dot(const BaseVector<NumType>& w) const = 0;

	/**
	 * Sets all values of the vector to zero.
	 */
	virtual void zeros() = 0;

	/**
	 * v = alpha*v, where v is this.
	 */
	virtual void scale(NumType alpha) = 0;

	/**
	 * v = alpha*v + beta*w, where v is this.
	 */
	virtual void add(NumType alpha, const BaseVector<NumType>& w, NumType beta) = 0;

	/**
	 * v = v + alpha*w, where v is this.
	 */
	virtual void add_scale(NumType alpha, const BaseVector<NumType>& w) = 0;

	/**
	 * v = alpha*v + w, where v is this.
	 */
	virtual void scale_add(NumType alpha, const BaseVector<NumType>& w) = 0;

protected:
	/**
	 * Size of the vector.
	 */
	int size_;

};
#endif