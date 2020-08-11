#include <stdio.h>
#include <cassert>
#include <cmath>

#include "solvers/gmres.h"
#include "solvers/base_preconditioner.h"

template <class MatType, class VecType, typename NumType>
GMRES<MatType, VecType, NumType>::GMRES()
    :   krylov_dim_(20),
        V_(nullptr)
{
}

template <class MatType, class VecType, typename NumType>
GMRES<MatType, VecType, NumType>::~GMRES()
{
}

template <class MatType, class VecType, typename NumType>
void GMRES<MatType, VecType, NumType>::clear() 
{
    if (krylov_dim_ >= 0) {
        c_.clear();
        s_.clear();
        g_.clear();
        H_.clear();
        
        delete[] V_;
    }
}

template <class MatType, class VecType, typename NumType>
void GMRES<MatType, VecType, NumType>::set_krylov_dimension(int K_dim) 
{
    krylov_dim_ = K_dim;
}

template <class MatType, class VecType, typename NumType>
void GMRES<MatType, VecType, NumType>::prepare_solver_(const MatType& mat)
{
    assert(krylov_dim_ > 0);
    clear();

    c_.allocate(krylov_dim_);
    s_.allocate(krylov_dim_);
    g_.allocate(krylov_dim_ + 1);
    H_.allocate((krylov_dim_ + 1)*krylov_dim_);

    V_ = new VecType[krylov_dim_ + 1];
    for (int i = 0; i < krylov_dim_ + 1; i++) {
        V_[i].allocate(mat.n());
    }

    if (this->precond_ != nullptr) {
        this->precond_->prepare_preconditioner(mat);
    }
}

template <class MatType, class VecType, typename NumType>
void GMRES<MatType, VecType, NumType>::solve(const MatType& mat, const VecType& rhs, VecType* soln)
{
    assert(mat.is_square());

    prepare_solver_(mat);

    this->it_counter_ = 0;

    const NumType one = static_cast<NumType>(1);
    const NumType minus_one = -one;
    const int nd = krylov_dim_;
    const int ndp1 = krylov_dim_ + 1;

    // V[0] = rhs - mat*x;
    mat.multiply(*soln, &V_[0]);
    V_[0].scale_add(minus_one, rhs);

    // V[0] <- M^-1 * V[0]
    if (this->precond_ != nullptr) {
        this->precond_->apply(&V_[0]);
    }

    g_.zeros();    
    g_[0] = V_[0].norm();
    V_[0].scale(one/g_[0]);
    this->init_res_norm_ = std::abs(g_[0]);
    //printf("init res: %e \n", this->init_res_norm_);
    while (not this->is_converged_()) {
        ++this->it_counter_;
        if (this->it_counter_ > this->max_its_) {
            break;
        }
    
        int j = 0;
        for (; j < nd; ++j) {
            // construct j-th orthonormal basis

            // v[j+1] = mat * v[j]
            mat.multiply(V_[j], &V_[j+1]);

            // v[j+1] <-- M^-1 * v[j+1]
            if (this->precond_ != nullptr) {
                this->precond_->apply(&V_[j+1]);
            }

            for (int i = 0; i < j + 1; i++) {
                H_[j*ndp1 + i] = V_[i].dot(V_[j+1]); // H[i,j] = V[i]^T*V[j+1]
                V_[j+1].add_scale(-H_[j*ndp1 + i], V_[i]); // V[j+1] -= H[i,j]*V[i]
            }
    
            H_[j*ndp1 + j+1] = V_[j+1].norm(); // H[j+1, j]
            V_[j+1].scale(one/H_[j*ndp1 + j+1]); // V[j+1] /= H[j+1, j]
    
            // factorizing H_j
            // apply previous rotations to new column
            for (int i = 0; i < j; i++) {
                GMRES<MatType, VecType, NumType>::rotate_inplace_(c_[i], s_[i], H_[j*ndp1 + i], H_[j*ndp1 + i+1]);
            }
            // construct new rotation
            NumType r = H_[j*ndp1 + j]; // H[j,j]
            NumType h = H_[j*ndp1 + j+1]; // H[j, j+1]
            NumType vinv = one/sqrt(r*r + h*h);
            c_[j] = r*vinv;
            s_[j] = -h*vinv;
    
            // apply new rotation to R[:,j] and g
            rotate_inplace_(c_[j], s_[j], H_[j*ndp1 + j], H_[j*ndp1 + j+1]);
            rotate_inplace_(c_[j], s_[j], g_[j], g_[j+1]);
    
            // have we converged?
            // if no, no need to solve for y
            this->res_norm_ = std::abs(g_[j+1]);

            //printf("i: %d j: %d r: %e \n", this->it_counter_, j, this->res_norm_);
            if (this->is_converged_()) {
                break;
            }
        }

        // solve for y (upper triangular)
        // write to g to save space
        for (int i = j - 1; i >= 0; i--) {
            g_[i] /= H_[i*ndp1 + i];
            for (int k = 0; k < i; k++) {
                g_[k] -= H_[i*ndp1 + k]*g_[i];
            }
        }

        soln->add_scale(g_[0], V_[0]);
        // update solution
        for (int i = 1; i < j; i++) {
            soln->add_scale(g_[i], V_[i]);
        }

        // prepare for next iteration
        mat.multiply(*soln, &V_[0]);
        V_[0].scale_add(minus_one, rhs);
        if (this->precond_ != nullptr) {
            this->precond_->apply(&V_[0]);
        }
        g_.zeros();
        g_[0] = V_[0].norm();
        V_[0].scale(one/g_[0]);
        //printf("i: %d r: %e \n", this->it_counter_,  this->res_norm_);
        this->res_norm_ = std::abs(g_[0]);
    }

    clear();
}

template <class MatType, class VecType, typename NumType>
void GMRES<MatType, VecType, NumType>::rotate_inplace_(NumType c, NumType s, NumType& h, NumType& hp1)
{
    NumType temp = h;
    h = c*h - s*hp1;
    hp1 = s*temp + c*hp1;
}

// instantiate template classes
template class GMRES<HostMatrix<double>, HostVector<double>, double>;
template class GMRES<HostMatrix<float>, HostVector<float>, float>;

#ifdef __CUDACC__
template class GMRES<DeviceMatrix<double>, DeviceVector<double>, double>;
template class GMRES<DeviceMatrix<float>, DeviceVector<float>, float>;
#endif
