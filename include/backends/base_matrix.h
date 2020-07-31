#ifndef BASE_MATRIX_H
#define BASE_MATRIX_H

#include "backends/base_vector.h"

// forward declaration
template <typename NumType>
class BaseVector;

template <typename NumType>
class BaseMatrix {
public:
	/**
	 * Constructor.
	 */
	BaseMatrix();

	/**
	 * Destructor.
	 */
	virtual ~BaseMatrix();

	/**
	 * Allocate a matrix of size m x n with nnz non-zero entries.
	 */
	virtual void allocate(int m, int n, int nnz) = 0;

	/**
	 * Clear matrix.
	 */
	virtual void clear() = 0;

	/**
	 * Copy data to the matrix.
	 * allocate() should be called first.
	 */
	virtual void copy(const NumType* val, const int* row_ptr, const int* col_idx) = 0;

	/**
	 * A = B, where A is this.
	 */
	virtual void copy(const BaseMatrix<NumType>& B) = 0;

	/** 
	 * Returns the row dimension of the matrix.
	 */
	int m() const;

	/** 
	 * Returns the column dimension of the matrix.
	 */
	int n() const;

	/**
	 * Returns true if matrix is square.
	 */
	bool is_square() const;

	/** 
	 * Returns the number of non-zero entries in the matrix.
	 */
	int nnz() const;

	/**
	 * Returns the Frobenius norm.
	 */
	virtual NumType norm() const = 0;

	/**
	 * A = alpha*A, where A is this.
	 */
	virtual void scale(NumType alpha) = 0;

	/**
	 * A = alpha*A + beta*B, where A is this.
	 */
	//virtual void add(NumType alpha, const BaseMatrix<NumType>& B, NumType beta) = 0;

	/**
	 * A = A + alpha*B, where A is this.
	 */
	//virtual void add_scale(NumType alpha, const BaseMatrix<NumType>& B) = 0;

	/**
	 * A = alpha*A + B, where A is this.
	 */
	//virtual void scale_add(NumType alpha, const BaseMatrix<NumType>& B) = 0;

	/**
	 * w_out = A*v_in, where A is this.
	 */
	virtual void multiply(const BaseVector<NumType>& v_in, 
		BaseVector<NumType>* w_out) const = 0;

protected:
	/**
	 * Row dimension of the matrix.
	 */
	int m_;

	/**
	 * Column dimension of the matrix.
	 */
	int n_;
	
	/**
	 * Number of non-zero entries in the matrix.
	 */
	int nnz_;

};
#endif