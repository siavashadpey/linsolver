#ifndef HOST_MATRIX_H
#define HOST_MATRIX_H

#include "backends/base_matrix.h"

// forward declaration
template <typename Number> 
class HostVector;

/**
 * \brief Implementation of a Matrix class on the host system.
 *
 * This class uses the Compressed Row Storage (CRS) method.
 */
template <typename NumType>
class HostMatrix: public BaseMatrix<NumType> {
public:
	/**
	 * Default constructor.
	 */
	HostMatrix();

	/**
	 * Destructor.
	 */
	virtual ~HostMatrix();

	virtual void allocate(int m, int n, int nnz);
	virtual void clear();
	virtual void copy(const NumType* val, const int* row_ptr, const int* col_idx);
	virtual void copy(const BaseMatrix<NumType>& B);

	virtual NumType norm() const;
	virtual void scale(NumType alpha);
	//virtual void add(NumType alpha, const BaseMatrix<NumType>& B, NumType beta);
	//virtual void add_scale(NumType alpha, const BaseMatrix<NumType>& B);
	//virtual void scale_add(NumType alpha, const BaseMatrix<NumType>& B);
	virtual void multiply(const BaseVector<NumType>& v_in, 
		BaseVector<NumType>* w_out) const;

protected:
	/**
	 * Values of non-zero entries in the matrix. (Compressed Row Storage)
	 */
	NumType* val_;

	/**
	 * Holds indices of val_ starting a new row. (Compressed Row Storage)
	 */
	NumType* row_ptr_;

	/**
	 * Column indices of the elements in val_. (Compressed Row Storage)
	 */
	NumType* col_idx_;
};
#endif