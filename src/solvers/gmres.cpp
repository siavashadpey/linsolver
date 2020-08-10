#include <stdio.h>
#include <cassert>
#include <cmath>

#include "solvers/gmres.h"
#include "backends/host/host_matrix.h"
#include "backends/host/host_vector.h"

template <typename NumType>
GMRES<NumType>::GMRES()
    :   krylov_dim_(20),
        V_(nullptr)
{
}

template <typename NumType>
GMRES<NumType>::~GMRES()
{
}

template <typename NumType>
void GMRES<NumType>::clear() 
{
    if (krylov_dim_ >= 0) {
        delete[] V_;
    }
}

template <typename NumType>
void GMRES<NumType>::set_krylov_dimension(int K_dim) 
{
    krylov_dim_ = K_dim;
}

template <typename NumType>
void GMRES<NumType>::prepare_solver_(int soln_dim)
{
    assert(krylov_dim_ > 0);
    clear();

    c_.allocate(krylov_dim_);
    s_.allocate(krylov_dim_);
    g_.allocate(krylov_dim_ + 1);
    H_.allocate((krylov_dim_ + 1)*krylov_dim_);

    V_ = new HostVector<NumType>[krylov_dim_ + 1];
    for (int i = 0; i < krylov_dim_ + 1; i++) {
        V_[i].allocate(soln_dim);
    }
}

template <typename NumType>
void GMRES<NumType>::solve(const BaseMatrix<NumType>& mat, const BaseVector<NumType>& rhs, BaseVector<NumType>* soln)
{
    assert(mat.is_square());

    // currently only handles host matrices and vectors
    const HostMatrix<NumType>* mat_h = dynamic_cast<const HostMatrix<NumType>*>(&mat);
    const BaseVector<NumType>* rhs_h = dynamic_cast<const HostVector<NumType>*>(&rhs);
    BaseVector<NumType>* soln_h = dynamic_cast<HostVector<NumType>*>(soln);
    assert(mat_h  != nullptr);
    assert(rhs_h  != nullptr);
    assert(soln_h != nullptr);

    prepare_solver_(mat_h->n());

    this->it_counter_ = 0;

    const NumType one = static_cast<NumType>(1);
    const NumType minus_one = -one;
    const int nd = krylov_dim_;
    const int ndp1 = krylov_dim_ + 1;

    // V[0] = rhs - mat*x;
    mat_h->multiply(*soln_h, &V_[0]);
    V_[0].scale_add(minus_one, *rhs_h);
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
            mat_h->multiply(V_[j], &V_[j+1]);
            for (int i = 0; i < j + 1; i++) {
                H_[j*ndp1 + i] = V_[i].dot(V_[j+1]); // H[i,j] = V[i]^T*V[j+1]
                V_[j+1].add_scale(-H_[j*ndp1 + i], V_[i]); // V[j+1] -= H[i,j]*V[i]
            }
    
            H_[j*ndp1 + j+1] = V_[j+1].norm(); // H[j+1, j]
            V_[j+1].scale(one/H_[j*ndp1 + j+1]); // V[j+1] /= H[j+1, j]
    
            // factorizing H_j
            // apply previous rotations to new column
            for (int i = 0; i < j; i++) {
                GMRES<NumType>::rotate_inplace_(c_[i], s_[i], H_[j*ndp1 + i], H_[j*ndp1 + i+1]);
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
            //printf("i: %d. j: %d. res: %e \n", this->it_counter_, j, g_[j+1]);
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

        soln_h->add_scale(g_[0], V_[0]);
        // update solution
        for (int i = 1; i < j; i++) {
            soln_h->add_scale(g_[i], V_[i]);
        }

        // prepare for next iteration
        mat_h->multiply(*soln_h, &V_[0]);
        V_[0].scale_add(minus_one, *rhs_h);    
        g_.zeros();
        g_[0] = V_[0].norm();
        V_[0].scale(one/g_[0]);
        this->res_norm_ = std::abs(g_[0]);
    }

    clear();
}

template <typename NumType>
void GMRES<NumType>::rotate_inplace_(NumType c, NumType s, NumType& h, NumType& hp1)
{
    NumType temp = h;
    h = c*h - s*hp1;
    hp1 = s*temp + c*hp1;
}

// instantiate template classes
template class GMRES<double>;
template class GMRES<float>;

// TODO: template MatType (HostMatrix and DeviceMatrix) and VecType (Hostvector and DeviceVector)
