#ifndef HOST_MATRIX_COO_H
#define HOST_MATRIX_COO_H

#include "backends/base_matrix.h"
#include "backends/host/host_matrix.h"

// forward declaration
template <typename NumType> 
class HostVector;
template <typename NumType> 
class HostMatrix;

/**
 * \brief Implementation of a Matrix class on the host system.
 * \tparam NumType Number type (double and float currently supported).
 * This class uses the Coordinate list (COO) format.
 */
template <typename NumType>
class HostMatrixCOO: public BaseMatrix<NumType> {
public:
    /**
     * Default constructor.
     */
    HostMatrixCOO();

    /**
     * Destructor.
     */
    ~HostMatrixCOO();

    void allocate(int m, int n, int nnz);
    void clear();

    void copy_from(const BaseMatrix<NumType>& B);
    void copy_to(BaseMatrix<NumType>& B) const;
    NumType norm() const;
    void scale(NumType alpha);
    void multiply(const BaseVector<NumType>& v_in, 
        BaseVector<NumType>* w_out) const;
    bool read_matrix_market(const std::string filename);

    /**
     * Convert the COO matrix to a CSR matrix.
     * @param[out] mat_CSR The CSR matrix.
     */
    void convert_to_CSR(HostMatrix<NumType>& mat_csr) const;

private:
    /**
     * Holds the row indices
     */
    int* row_idx_;

    /**
     * Holds the column indices
     */
    int* col_idx_;

    /**
     * Holds the values of the non-zero entries of the matrix
     */
    NumType *val_;
};
#endif
