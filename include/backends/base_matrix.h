#ifndef BASE_MATRIX_H
#define BASE_MATRIX_H

#include "backends/base_vector.h"

// forward declaration
template <typename NumType>
class BaseVector;

/**
 * \brief Base matrix class of HostMatrix and DeviceMatrix classes.
 */
template <typename NumType>
class BaseMatrix {
public:
	/**
	 * Default constructor.
	 */
	BaseMatrix();

	/**
	 * Destructor.
	 */
	virtual ~BaseMatrix();

	/**
	 * Allocate a matrix of size m x n with nnz non-zero entries.
	 * @param[in] m The number of rows.
	 * @param[in] n The number of columns.
	 * @param[in] nnz The number of non-zero entries.
	 */
	virtual void allocate(int m, int n, int nnz) = 0;

	/**
	 * Clear matrix.
	 */
	virtual void clear() = 0;

	/**
	 * Copy data to the matrix.
	 * @param[in] val The pointer to the values to copy.
	 * @param[in] row_ptr The pointer to the indices of val starting a new row.
	 * @param[in] col_idx The pointer to the column indices of the elements in val.
	 * @note allocate should be called first.
	 */
	virtual void copy(const NumType* val, const int* row_ptr, const int* col_idx) = 0;

	/**
	 * A = B, where A is the matrix itself.
	 * @param[in] B The matrix to copy.
	 */
	virtual void copy(const BaseMatrix<NumType>& B) = 0;

	/** 
	 * \return The row dimension of the matrix.
	 */
	int m() const;

	/** 
	 * \return The column dimension of the matrix.
	 */
	int n() const;

	/** 
	 * \return The number of non-zero entries in the matrix.
	 */
	int nnz() const;

	/**
	 * \return True if the matrix is square and false if it is not.
	 */
	bool is_square() const;

	/**
	 * \return The Frobenius norm of the matrix.
	 */
	virtual NumType norm() const = 0;

	/**
	 * A = alpha*A, where A is the matrix itself.
	 * \param[in] alpha The value by which to scale the entries in the matrix.
	 */
	virtual void scale(NumType alpha) = 0;

	/**
	 * A = alpha*A + beta*B, where A is the matrix itself.
	 * \param[in] alpha The value in the above equation.
	 * \param[in] B     The matrix in the above equation.
	 * \param[in] beta  The value in the above equation.
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
	 * w_out = A*v_in, where A is the matrix itself.
	 * \param[in] v_in The vector right-multiplying the vector.
	 * \param[out] w_out The outputted vector.
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