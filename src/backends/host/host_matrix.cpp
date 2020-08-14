#include <stdio.h>
#include <cmath>
#include <cassert>

#include "backends/host/host_matrix.h"
#include "backends/host/host_vector.h"

template <typename NumType>
HostMatrix<NumType>::HostMatrix()
    :   val_(nullptr),
        row_ptr_(nullptr),
        col_idx_(nullptr)
{
}

template <typename NumType>
HostMatrix<NumType>::~HostMatrix()
{
    this->clear();
}

template <typename NumType>
void HostMatrix<NumType>::allocate(int m, int n, int nnz)
{
    assert(nnz > 0);
    this->clear();

    this->m_ = m;
    this->n_ = n;
    this->nnz_ = nnz;

    this->val_ = new NumType[nnz]();
    this->row_ptr_ = new int[m+1]();
    this->col_idx_ = new int[nnz]();
}

template <typename NumType>
void HostMatrix<NumType>::clear()
{
    if (this->nnz_ > 0) {
        delete[] this->val_;
        delete[] this->row_ptr_;
        delete[] this->col_idx_;
        this->m_ = 0;
        this->n_ = 0;
        this->nnz_ = 0;
    }
}

template <typename NumType>
void HostMatrix<NumType>::copy_from(const NumType* val, const int* row_ptr, const int* col_idx)
{
    assert(this->m_ > 0);
    assert(this->n_ > 0);
    assert(this->nnz_ > 0);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < this->nnz_; i++) {
        this->val_[i] = val[i];
        this->col_idx_[i] = col_idx[i];
    }

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < this->m_ + 1; i++) {
        this->row_ptr_[i] = row_ptr[i];
    }
}

template <typename NumType>
void HostMatrix<NumType>::copy_from(const BaseMatrix<NumType>& B)
{
    assert(this != &B);
    const HostMatrix<NumType>* B_host = dynamic_cast<const HostMatrix<NumType>*>(&B);
    assert(B_host != nullptr);

    if ((B_host->nnz_ != this->nnz_) or 
        (B_host->m_ != this->m_) or
        (B_host->n_ != this->n_)) {
            this->allocate(B_host->m_, B_host->n_, B_host->nnz_);
    }

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < this->nnz_; i++) {
        this->val_[i] = B_host->val_[i];
        this->col_idx_[i] = B_host->col_idx_[i];
    }

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < this->m_ + 1; i++) {
        this->row_ptr_[i] = B_host->row_ptr_[i];
    }
}

template <typename NumType>
void HostMatrix<NumType>::copy_to(BaseMatrix<NumType>& B) const
{
    B.copy_from(*this);
}

template <typename NumType>
NumType HostMatrix<NumType>::norm() const
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
void HostMatrix<NumType>::scale(NumType alpha)
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

//template <typename NumType>
//void HostMatrix<NumType>::add(NumType alpha, const BaseMatrix<NumType>& B, NumType beta)
//{
//}
//
//template <typename NumType>
//void HostMatrix<NumType>::add_scale(NumType alpha, const BaseMatrix<NumType>& B)
//{
//}
//
//template <typename NumType>
//void HostMatrix<NumType>::scale_add(NumType alpha, const BaseMatrix<NumType>& B)
//{
//}

template <typename NumType>
void HostMatrix<NumType>::multiply(const BaseVector<NumType>& v_in, 
        BaseVector<NumType>* w_out) const
{
    // check dimensions match
    assert(this->n_ > 0);
    assert(this->m_ > 0);
    assert(v_in.n() == this->n_);
    assert(w_out->n() == this->m_);

    const HostVector<NumType>* v_in_h = dynamic_cast<const HostVector<NumType>*>(&v_in);    
    HostVector<NumType>* w_out_h = dynamic_cast<HostVector<NumType>*>(w_out);
    assert(v_in_h != nullptr);
    assert(w_out_h != nullptr);

    const NumType zero = static_cast<NumType>(0);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int irow = 0; irow < this->m_; irow++) {
        NumType val_i = zero;
        for (int ij = this->row_ptr_[irow]; ij < this->row_ptr_[irow+1]; ij++) {
            int jcol = this->col_idx_[ij];
            val_i += this->val_[ij]*v_in_h->vec_[jcol];
        }
        w_out_h->vec_[irow] = val_i;
    }
}

template <typename NumType>
void HostMatrix<NumType>::lower_solve(const BaseVector<NumType>& b, BaseVector<NumType>* x) const
{
    assert(this->n_ > 0);
    assert(this->m_ > 0);
    assert(b.n() == this->n_);
    assert(x->n() == this->m_);

    const HostVector<NumType>* b_h = dynamic_cast<const HostVector<NumType>*>(&b);    
    HostVector<NumType>* x_h = dynamic_cast<HostVector<NumType>*>(x);
    assert(b_h != nullptr);
    assert(x_h != nullptr);

    for (int irow = 0; irow < this->m_; irow++) {
        NumType sum = b_h->vec_[irow];
        for (int ij = this->row_ptr_[irow]; ij < this->row_ptr_[irow+1]; ij++) {
            int jcol = this->col_idx_[ij];
            if (jcol < irow) {
                sum -= this->val_[ij]*x_h->vec_[jcol];
            }
            else { // jcol == irow
                x_h->vec_[irow] = sum/this->val_[ij];
                break;
            }
        }
        
    }
}

template <typename NumType>
void HostMatrix<NumType>::upper_solve(const BaseVector<NumType>& b, BaseVector<NumType>* x) const
{
    assert(this->n_ > 0);
    assert(this->m_ > 0);
    assert(b.n() == this->n_);
    assert(x->n() == this->m_);

    const HostVector<NumType>* b_h = dynamic_cast<const HostVector<NumType>*>(&b);    
    HostVector<NumType>* x_h = dynamic_cast<HostVector<NumType>*>(x);
    assert(b_h != nullptr);
    assert(x_h != nullptr);

    for (int irow = this->m_ - 1; irow >= 0; irow--) {
        NumType sum = b_h->vec_[irow];
        for (int ij = this->row_ptr_[irow+1] - 1; ij >= this->row_ptr_[irow]; ij--) {
            int jcol = this->col_idx_[ij];
            if (jcol > irow) {
                sum -= this->val_[ij]*x_h->vec_[jcol];
            }
            else { // jcol == irow
                x_h->vec_[irow] = sum/this->val_[ij];
                break;
            }
        }
    }
}

template <typename NumType>
void HostMatrix<NumType>::lower_upper_solve(const BaseVector<NumType>& b, BaseVector<NumType>* x) const
{
    
    assert(this->n_ > 0);
    assert(this->m_ > 0);
    assert(b.n() == this->n_);
    assert(x->n() == this->m_);

    const HostVector<NumType>* b_h = dynamic_cast<const HostVector<NumType>*>(&b);    
    HostVector<NumType>* x_h = dynamic_cast<HostVector<NumType>*>(x);
    assert(b_h != nullptr);
    assert(x_h != nullptr);

    // Solve the lower part (forward substitution)
    for (int irow = 0; irow < this->m_; irow++) {
        x_h->vec_[irow] = b_h->vec_[irow];
        for (int ij = this->row_ptr_[irow]; ij < this->row_ptr_[irow+1]; ij++) {
            int jcol = this->col_idx_[ij];
            if (jcol >= irow) {
               break;
            }
            x_h->vec_[irow] -= this->val_[ij]*x_h->vec_[jcol];
        }
    }

    // Solve the upper part (backward substitution)
    for (int irow = this->m_ - 1; irow >= 0; irow--) {
        for (int ij = this->row_ptr_[irow+1] - 1; ij >= this->row_ptr_[irow]; ij--) {
            int jcol = this->col_idx_[ij];
            if (jcol > irow) {
                x_h->vec_[irow] -= this->val_[ij]*x_h->vec_[jcol];
            }
            else { // (jcol == irow)
                x_h->vec_[irow] /= this->val_[ij];
                break;
            }
        }
    }
}

template <typename NumType>
void HostMatrix<NumType>::get_diagonals(BaseVector<NumType>* diag) const
{
    assert(this->n_ > 0);
    assert(this->m_ > 0);
    assert(diag != nullptr);

    HostVector<NumType>* diag_h = dynamic_cast<HostVector<NumType>*>(diag);
    assert(diag_h != nullptr);

    if (diag_h->n() != this->m_) {
            diag_h->allocate(this->m_);
    }

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int irow = 0; irow < this->m_; irow++) {
        for (int ij = this->row_ptr_[irow]; ij < this->row_ptr_[irow+1]; ij++) {
            int jcol = this->col_idx_[ij];
            if (irow == jcol) {
                diag_h->vec_[irow] = this->val_[ij];
                break;
            }
        }
    }
}

template <typename NumType>
void HostMatrix<NumType>::compute_inverse_diagonals(BaseVector<NumType>* inv_diag) const
{
    assert(this->n_ > 0);
    assert(this->m_ > 0);
    assert(inv_diag != nullptr);

    HostVector<NumType>* inv_diag_h = dynamic_cast<HostVector<NumType>*>(inv_diag);
    assert(inv_diag_h != nullptr);

    const NumType one = static_cast<NumType>(1);

    if (inv_diag_h->n() != this->m_) {
            inv_diag_h->allocate(this->m_);
    }

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int irow = 0; irow < this->m_; irow++) {
        for (int ij = this->row_ptr_[irow]; ij < this->row_ptr_[irow+1]; ij++) {
            int jcol = this->col_idx_[ij];
            if (irow == jcol) {
                inv_diag_h->vec_[irow] = one/this->val_[ij];
                break;
            }
        }
    }
}

template <typename NumType>
void HostMatrix<NumType>::ILU0_factorize()
{
    // from SAAD 03 Chapter 10.3

    assert(this->n_ > 0);
    assert(this->m_ = this->n_);

    NumType zero = static_cast<NumType>(0);
    
    int* diag_ptr = new int[this->n_]();
    int* irow_nnz_idx = new int[this->n_]();

    // loop through each row
    for (int irow = 0; irow < this->m_; irow++) {
        
        // set indices of nonzero entries in current row
        for (int ik = this->row_ptr_[irow]; ik < this->row_ptr_[irow+1]; ik++) {
            irow_nnz_idx[this->col_idx_[ik]] = ik;
        }

        // loop through all nonzero entries in current row until diagonal entry (i.e. A[k,i] for k < i)
        int ik = this->row_ptr_[irow];
        for (; ik < this->row_ptr_[irow+1]; ik++) {
            int kcol = this->col_idx_[ik];
            if (kcol >= irow) {
                break;
            }

            if (this->val_[diag_ptr[kcol]] != zero) {
                this->val_[ik] /= this->val_[diag_ptr[kcol]]; // A(i, k) = A(i, k) / A(k, k)
    
                // loop through A[irow, j] for all j > k
                for (int kj = diag_ptr[kcol] + 1; kj < this->row_ptr_[kcol+1]; kj++) {
                    int jcol = this->col_idx_[kj];
                    int ij = irow_nnz_idx[jcol];
                    if (ij != 0) {
                        this->val_[ij] -= this->val_[ik] * this->val_[kj];
                    }
                }
            }
        }
        diag_ptr[irow] = ik;

        // reset nnz indices for next row
        for (ik = this->row_ptr_[irow]; ik < this->row_ptr_[irow+1]; ik++) {
            irow_nnz_idx[this->col_idx_[ik]] = 0;
        }
    }

    delete[] diag_ptr;
    delete[] irow_nnz_idx;
}

template <typename NumType>
bool HostMatrix<NumType>::read_matrix_market(const std::string filename) {
    HostMatrixCOO<NumType> mat_coo = HostMatrixCOO<NumType>();
    bool success = mat_coo.read_matrix_market(filename);
    if (success) {
        mat_coo.convert_to_CSR(*this);
    }
    return success;
}

// instantiate template classes
template class HostMatrix<double>;
template class HostMatrix<float>;
