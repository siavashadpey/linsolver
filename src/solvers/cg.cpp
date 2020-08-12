#include <stdio.h>
#include <cassert>
#include <cmath>

#include "solvers/cg.h"
#include "solvers/base_preconditioner.h"

template <class MatType, class VecType, typename NumType>
CG<MatType, VecType, NumType>::CG()
    :   z_(nullptr)
{
}

template <class MatType, class VecType, typename NumType>
CG<MatType, VecType, NumType>::~CG()
{
}

template <class MatType, class VecType, typename NumType>
void CG<MatType, VecType, NumType>::clear() 
{
    r_.clear();
    p_.clear();
    q_.clear();

    if (this->precond_ != nullptr and z_ != nullptr) {
        delete z_;
    }
}

template <class MatType, class VecType, typename NumType>
void CG<MatType, VecType, NumType>::prepare_solver_(const MatType& mat)
{
    clear();

    int soln_dim = mat.n();
    
    r_.allocate(soln_dim);
    p_.allocate(soln_dim);
    q_.allocate(soln_dim);
    
    if (this->precond_ != nullptr) {
        z_ = new VecType;
        z_->allocate(soln_dim);

        this->precond_->prepare_preconditioner(mat);
    }
    else {
        z_ = &r_;
    }
}

template <class MatType, class VecType, typename NumType>
void CG<MatType, VecType, NumType>::solve(const MatType& mat, const VecType& rhs, VecType* soln)
{
    assert(mat.is_square());

    prepare_solver_(mat);

    const NumType zero = static_cast<NumType>(0);
    const NumType one = static_cast<NumType>(1);
    const NumType minus_one = -one;

    // r = rhs - mat*x
    mat.multiply(*soln, &r_);
    r_.scale_add(minus_one, rhs);

    NumType rho = zero;
    if (this->precond_ != nullptr) {
        // solve for z in M*z = r
        this->precond_->apply(r_, z_);
        rho = r_.dot(*z_);
        this->init_res_norm_ = r_.norm();
    }
    else {
        rho = r_.dot(r_);
        this->init_res_norm_ = sqrt(rho);
    }

    //printf("init res: %e \n", this->init_res_norm_);

    NumType beta = zero;
    for (this->it_counter_ = 0; this->it_counter_ < this->max_its_; ++this->it_counter_) {
        p_.scale_add(beta, *z_); // p = z + beta * p

        mat.multiply(p_, &q_); // q = A*p
        NumType alpha = rho / (q_.dot(p_));

        soln->add_scale(alpha, p_); // x = x + alpha * p
        r_.add_scale(-alpha, q_);   // r = r - alpha * q

        this->res_norm_ = r_.norm();
        //printf("i: %d r: %e \n", this->it_counter_,  this->res_norm_);
        if (this->is_converged_()) {
            break;
        }

        if (this->precond_ != nullptr) {
            // solve for z in M*z = r
            this->precond_->apply(r_, z_);
        }

        // prepara for next iteration
        beta = one/rho;
        if (this->precond_ != nullptr) {
            rho = r_.dot(*z_);
        }
        else {
            rho = this->res_norm_*this->res_norm_;
        }
        beta = beta * rho;
    }
    clear();
}

// instantiate template classes
template class CG<HostMatrix<double>, HostVector<double>, double>;
template class CG<HostMatrix<float>, HostVector<float>, float>;

#ifdef __CUDACC__
template class CG<DeviceMatrix<double>, DeviceVector<double>, double>;
template class CG<DeviceMatrix<float>, DeviceVector<float>, float>;
#endif
