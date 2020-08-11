#include <fstream>
#include <cmath>
#include <cstring>

#include "base/error.h"
#include "backends/host/host_matrix_coo.h"
#include "backends/host/host_vector.h"

template <typename NumType>
HostMatrixCOO<NumType>::HostMatrixCOO()
{
}

template <typename NumType>
HostMatrixCOO<NumType>::~HostMatrixCOO()
{
    this->clear();
}

template <typename NumType>
void HostMatrixCOO<NumType>::allocate(int m, int n, int nnz)
{
    assert(nnz > 0);
    this->clear();

    this->m_ = m;
    this->n_ = n;
    this->nnz_ = nnz;

    this->row_idx_ = new int[nnz];
    this->col_idx_ = new int[nnz];
    this->val_ = new NumType[nnz];
}

template <typename NumType>
void HostMatrixCOO<NumType>::clear()
{
    if (this->nnz_ > 0) {
        delete[] this->row_idx_;
        delete[] this->col_idx_;
        delete[] this->val_;
        this->m_ = 0;
        this->n_ = 0;
        this->nnz_ = 0;
    }
}

template <typename NumType>
void HostMatrixCOO<NumType>::copy_from(const BaseMatrix<NumType>& )
{
    Error("Method has not yet been implemented.");
}

template <typename NumType>
void HostMatrixCOO<NumType>::copy_to(BaseMatrix<NumType>& ) const 
{
    Error("Method has not yet been implemented.");
}

template <typename NumType>
NumType HostMatrixCOO<NumType>::norm() const
{
    NumType val = static_cast<NumType>(0);

#ifdef _OPENMP
    #pragma omp parallel for reduction(+:val)
#endif
    for (int i = 0; i < this->nnz_; i++) {
        val += this->val_[i]*this->val_[i];
    }
    return sqrt(val);
}

template <typename NumType>
void HostMatrixCOO<NumType>::scale(NumType alpha)
{
    const NumType one = static_cast<NumType>(1);

    if (alpha == one){
        return;
    }

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < this->nnz_; i++) {
        this->val_[i] *= alpha;
    }
}

template <typename NumType>
void HostMatrixCOO<NumType>::multiply(const BaseVector<NumType>& v_in, 
        BaseVector<NumType>* w_out) const
{
    assert(this->n_ > 0);
    assert(this->m_ > 0);
    assert(v_in.n() == this->n_);
    assert(w_out->n() == this->m_);

    const HostVector<NumType>* v_in_h = dynamic_cast<const HostVector<NumType>*>(&v_in);    
    HostVector<NumType>* w_out_h = dynamic_cast<HostVector<NumType>*>(w_out);
    assert(v_in_h != nullptr);
    assert(w_out_h != nullptr);

    NumType zero = static_cast<NumType>(0);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < this->m_; i++) {
        w_out_h->vec_[i] = zero;
    }

    for (int n = 0; n < this->nnz_; n++) {
        int irow = row_idx_[n];
        int jcol = col_idx_[n];
        w_out_h->vec_[irow] += val_[n]*v_in_h->vec_[jcol];    
    }
}

template <typename NumType>
void HostMatrixCOO<NumType>::compute_inverse_diagonals(BaseVector<NumType>* inv_diag) const
{
    Error("Method has not yet been implemented.");
}


template <typename NumType>
bool HostMatrixCOO<NumType>::read_matrix_market(const std::string filename)
{
    FILE* file = fopen(filename.c_str(), "r");
    if (!file) {
        return false;
    }

    char line[1024];

    //read banner
    if (!fgets(line, 1024, file)) {
        return false;
    }
    char temp[64];
    char format_type[64];
    char mat_type[64];
    char symm_type[64];
    sscanf(line, "%s %s %s %s %s", temp, temp, format_type, mat_type, symm_type);

    // make sure matrix format type is coordinate
    // and matrix type is real, double, or integer
    // and symm_type is general
    if (std::strcmp(format_type, "coordinate")) {
        return false;
    }
    if (std::strcmp(mat_type, "real") and
        std::strcmp(mat_type, "double") and 
        std::strcmp(mat_type, "integer")) {
        return false;
    }
    if (std::strcmp(symm_type, "general")) {
        return false;
    }

    // skip comments
    do
    {
        if (!fgets(line, 1024, file)) {
            return false;
        }
    } while (line[0] == '%'); 

    // read matrix dimensions and nnz
    int m;
    int n;
    int nnz;
    sscanf(line, "%d %d %d", &m, &n, &nnz);
    allocate(m, n, nnz);
    
    // read values
    for (int i = 0; i < this->nnz_; i++) {
        if (!fgets(line, 1024, file)) {
            return false;
        }
        double tmp_val;
        sscanf(line, "%d %d %lg", (row_idx_ + i), (col_idx_ + i), &tmp_val);
        --row_idx_[i]; // convert to zero-based
        --col_idx_[i]; // convert to zero-based
        val_[i] = static_cast<NumType>(tmp_val);
    }

    // successfully read file
    return true;
}

template <typename NumType>
void HostMatrixCOO<NumType>::convert_to_CSR(HostMatrix<NumType>& mat_csr) const
{
    mat_csr.allocate(this->m_, this->n_, this->nnz_);
    
    int* csr_row_ptr;
    csr_row_ptr = (int*) malloc((this->m_ + 1) * sizeof(int));
    std::fill(csr_row_ptr, csr_row_ptr + this->m_, 0);

    // count number of non-zero entries per row
    for (int i = 0; i < this->nnz_; i++) {
        ++csr_row_ptr[this->row_idx_[i]];
    }

    // row_ptr[irow] = cumsum of number of non-zero enrtries until irow-1
    int cumsum = 0;
    for (int i = 0; i < this->m_; i++) {
        int temp = csr_row_ptr[i];
        csr_row_ptr[i] = cumsum;
        cumsum += temp;
    }
    csr_row_ptr[this->m_] = this->nnz_;
    
    int* csr_col_idx;
    NumType* csr_val;
    csr_col_idx = (int*) malloc((this->nnz_) * sizeof(int));
    csr_val = (NumType*) malloc((this->nnz_) * sizeof(NumType));

    // in case row and columns of COO matrix are not ordered
    for (int i = 0; i < this->nnz_; i++) {
        int irow = this->row_idx_[i];
        int csr_idx = csr_row_ptr[irow];

        csr_val[csr_idx] = this->val_[i];
        csr_col_idx[csr_idx] = this->col_idx_[i];
        ++csr_row_ptr[irow];
    }
    int last = 0;
    for (int i = 0; i < this->m_ + 1; i++) {
        int temp = csr_row_ptr[i];
        csr_row_ptr[i] = last;
        last = temp;
    }

    mat_csr.copy_from(csr_val, csr_row_ptr, csr_col_idx);

    free(csr_row_ptr);
    free(csr_col_idx);
    free(csr_val);
}

// instantiate template classes
template class HostMatrixCOO<double>;
template class HostMatrixCOO<float>;
