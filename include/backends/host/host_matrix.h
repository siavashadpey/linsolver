#ifndef HOST_MATRIX_H
#define HOST_MATRIX_H

#include "backends/base_matrix.h"
#include "backends/host/host_matrix_coo.h"

#ifdef __CUDACC__
#include "backends/device/device_matrix.h"
#endif

// forward declaration
template <typename NumType> 
class HostVector;

/**
 * \brief Implementation of a Matrix class on the host system.
 * \tparam NumType Number type (double and float currently supported).
 * This class uses the Compressed Row Storage (CRS) format.
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

    /**
     * Copy data to the matrix.
     * @param[in] val The pointer to the values to copy.
     * @param[in] row_ptr The pointer to the indices of val starting a new row.
     * @param[in] col_idx The pointer to the column indices of the elements in val.
     * @note allocate should be called first.
     */
    virtual void copy_from(const NumType* val, const int* row_ptr, const int* col_idx);
    virtual void copy_from(const BaseMatrix<NumType>& B);

    virtual void copy_to(BaseMatrix<NumType>& B) const;

    virtual NumType norm() const;
    virtual void scale(NumType alpha);
    //virtual void add(NumType alpha, const BaseMatrix<NumType>& B, NumType beta);
    //virtual void add_scale(NumType alpha, const BaseMatrix<NumType>& B);
    //virtual void scale_add(NumType alpha, const BaseMatrix<NumType>& B);
    virtual void multiply(const BaseVector<NumType>& v_in, 
        BaseVector<NumType>* w_out) const;
    virtual void lower_solve(const BaseVector<NumType>& b, BaseVector<NumType>* x) const;
    virtual void upper_solve(const BaseVector<NumType>& b, BaseVector<NumType>* x) const;
    virtual void get_diagonals(BaseVector<NumType>* diag) const;
    virtual void compute_inverse_diagonals(BaseVector<NumType>* inv_diag) const;
    bool read_matrix_market(const std::string filename);

protected:
    /**
     * Values of non-zero entries in the matrix. (Compressed Row Storage)
     */
    NumType* val_;

    /**
     * Holds indices of val_ starting a new row. (Compressed Row Storage)
     */
    int* row_ptr_;

    /**
     * Column indices of the elements in val_. (Compressed Row Storage)
     */
    int* col_idx_;

    // befriending classes
private:
#ifdef __CUDACC__
    friend class DeviceMatrix<NumType>;
#endif
};
#endif
