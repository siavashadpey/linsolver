#ifndef BASE_MATRIX_H
#define BASE_MATRIX_H

#include <string>

#include "backends/base_vector.h"

// forward declaration
template <typename NumType>
class BaseVector;

/**
 * \brief Base matrix class.
 * \tparam NumType Number type (double and float currently supported).
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
     * Allocate a matrix of size \p m x \p n with \p nnz non-zero entries.
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
     * \p A = \p B, where \p A is the matrix itself.
     * @param[in] B The matrix to copy.
     */
    virtual void copy_from(const BaseMatrix<NumType>& B) = 0;

    /**
     * \p B = \p A, where \p A is the matrix itself.
     * @param[out] B The matrix to copy to.
     */
    virtual void copy_to(BaseMatrix<NumType>& B) const = 0;
    
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
     * \p A = \p alpha * \p A, where \p A is the matrix itself.
     * \param[in] alpha The value by which to scale the entries in the matrix.
     */
    virtual void scale(NumType alpha) = 0;

    //virtual void add(NumType alpha, const BaseMatrix<NumType>& B, NumType beta) = 0;
    //virtual void add_scale(NumType alpha, const BaseMatrix<NumType>& B) = 0;
    //virtual void scale_add(NumType alpha, const BaseMatrix<NumType>& B) = 0;

    /**
     * \p w_out = \p A * \p v_in, where \p A is the matrix itself.
     * \param[in] v_in The vector right-multiplying the vector.
     * \param[out] w_out The outputted vector.
     */
    virtual void multiply(const BaseVector<NumType>& v_in, 
        BaseVector<NumType>* w_out) const = 0;

    /**
     * Outputs the diagonal entries of the matrix.
     * \param[out] w_out The outputted vector storing the diagonal entries.
     */
    virtual void get_diagonals(BaseVector<NumType>* diag) const = 0;

    /**
     * Compute the inverse of the diagonal entries of the matrix.
     * \param[out] w_out The outputted vector storing the inverse of the diagonal entries.
     */
    virtual void compute_inverse_diagonals(BaseVector<NumType>* inv_diag) const = 0;

    /**
     * Solvers for \p x in (\p L + \p D) * \p x = \p b, 
     * where \p L and \p D are respectively the strictly lower triangular and diagonal parts of the matrix itself.
     * @\param[in] b The vector in the above equation
     * @\param[out] x The vector in the above equation
     */
    virtual void lower_solve(const BaseVector<NumType>& b, BaseVector<NumType>* x) const = 0;

    /**
     * Solvers for \p x in (\p U + \p D) * \p x = \p b, 
     * where \p U and \p D are respectively the stricly upper triangular and diagonal parts of the matrix itself.
     * @\param[in] b The vector in the above equation
     * @\param[out] x The vector in the above equation
     */
    virtual void upper_solve(const BaseVector<NumType>& b, BaseVector<NumType>* x) const = 0;

    /**
     * Read a matrix market.
     * @param[in] filename The matrix market file.
     * \returns True if successfully read, and False otherwise.
     */
    virtual bool read_matrix_market(const std::string filename) = 0;

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
