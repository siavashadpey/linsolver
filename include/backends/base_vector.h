#ifndef BASE_VECTOR_H
#define BASE_VECTOR_H

/**
 * \brief Base vector class of HostVector and DeviceVector classes.
 */
template <typename NumType>
class BaseVector {
public:
	/**
	 * Default onstructor.
	 */
	BaseVector();

	/**
	 * Destructor.
	 */
	virtual ~BaseVector();

	/**
	 * Allocate a size of \p n to the vector.
	 * @param[in] n The size of vector.
	 */
	virtual void allocate(int n) = 0;

	/**
	 * Clear vector.
	 */
	virtual void clear() = 0;

	/**
	 * Copy data to the vector.
	 * @note allocate should be called first.
	 * @param[in] w The pointer to the data to copy.
	 */
	virtual void copy(const NumType* w) = 0;

	/**
	 * Copy a vector to itself.
	 * @param[in] w The vector to copy.
	 */
	virtual void copy(const BaseVector<NumType>& w) = 0;

	/** 
	 * \return The size of the vector.
	 */
	int n() const;

	/**
	 * \return The L2 norm of the vector.
	 */
	virtual NumType norm() const = 0;

	/**
	 * Return the dot product between itself and another vector.
	 * \param[in] w The other vector
	 * \return The dot product.
	 */
	virtual NumType dot(const BaseVector<NumType>& w) const = 0;

	/**
	 * Sets all values of the vector to zero.
	 */
	virtual void zeros() = 0;

	/**
	 * Scale the vector.
	 * \param[in] alpha the value by which to scale the vector.
	 */
	virtual void scale(NumType alpha) = 0;

	/**
	 * \p v = \p alpha * \p v + \p beta * \p w, where \p v is the vector itself.
	 * \param[in] alpha The value in the above equation.
	 * \param[in] w     The vector in the above equation.
	 * \param[in] beta  The value in the above equation.
	 */
	virtual void add(NumType alpha, const BaseVector<NumType>& w, NumType beta) = 0;

	/**
	 * \p v = \p v + \p alpha * \p w, where \p v is the vector itself.
	 * \param[in] alpha The value in the above equation.
	 * \param[in] w     The vector in the above equation.
	 */
	virtual void add_scale(NumType alpha, const BaseVector<NumType>& w) = 0;

	/**
	 * \p v = \p alpha * \p v + \p w, where \p v is the vector itself.
	 * \param[in] alpha The value in the above equation.
	 * \param[in] w     The vector in the above equation.
	 */
	virtual void scale_add(NumType alpha, const BaseVector<NumType>& w) = 0;

protected:
	/**
	 * Size of the vector.
	 */
	int size_;
};
#endif