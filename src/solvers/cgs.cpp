#include <stdio.h>
#include <cassert>
#include <cmath>

#include "base/error.h"

#include "solvers/cgs.h"
#include "solvers/base_preconditioner.h"

template <class MatType, class VecType, typename NumType>
CGS<MatType, VecType, NumType>::CGS()
    :   p_hat_(nullptr)
{
}

template <class MatType, class VecType, typename NumType>
CGS<MatType, VecType, NumType>::~CGS()
{
}

template <class MatType, class VecType, typename NumType>
void CGS<MatType, VecType, NumType>::clear() 
{
    r_.clear();
    q_.clear();
    p_.clear();
    u_.clear();
    r_hat_.clear();
    u_hat_.clear();
    v_hat_.clear();
    q_hat_.clear();

    if (this->precond_ != nullptr and p_hat_ != nullptr) {
        delete p_hat_;
    }
}

template <class MatType, class VecType, typename NumType>
void CGS<MatType, VecType, NumType>::prepare_solver_(const MatType& mat)
{
    clear();

    int soln_dim = mat.n();

    r_.allocate(soln_dim);
    q_.allocate(soln_dim);
    p_.allocate(soln_dim);
    u_.allocate(soln_dim);
    r_hat_.allocate(soln_dim);
    u_hat_.allocate(soln_dim);
    v_hat_.allocate(soln_dim);
    q_hat_.allocate(soln_dim);
    
    if (this->precond_ != nullptr) {
        p_hat_ = new VecType;
        p_hat_->allocate(soln_dim);
        this->precond_->prepare_preconditioner(mat);
    }
    else {
        p_hat_ = &p_;
    }
}

template <class MatType, class VecType, typename NumType>
void CGS<MatType, VecType, NumType>::solve(const MatType& mat, const VecType& rhs, VecType* soln)
{
    assert(mat.is_square());

    prepare_solver_(mat);

    const NumType zero = static_cast<NumType>(0);
    const NumType one = static_cast<NumType>(1);
    const NumType minus_one = -one;

    // r = rhs - mat*x
    mat.multiply(*soln, &r_);
    r_.scale_add(minus_one, rhs);

    r_hat_.copy_from(r_);

    NumType rho = r_hat_.dot(r_);

    this->init_res_norm_ = sqrt(rho);
    //printf("init res: %e \n", this->init_res_norm_);

    NumType beta = zero;
    for (this->it_counter_ = 0; this->it_counter_ < this->max_its_; ++this->it_counter_) {
        
        // u = r + beta * q
        u_.copy_from(r_);
        u_.add_scale(beta, q_);

        // p = u + beta*( beta*p + q )
        p_.scale_add(beta, q_);
        p_.scale_add(beta, u_);

        if (this->precond_ != nullptr) {
            // solve for z in M*p_hat = p
            this->precond_->apply(p_, p_hat_);
        }
        mat.multiply(*p_hat_, &v_hat_); // v_hat = A*p_hat

        NumType alpha = rho/(v_hat_.dot(r_hat_));

        // q = u - alpha * v_hat
        q_.copy_from(u_);
        q_.add_scale(-alpha, v_hat_);

        u_hat_.copy_from(u_);
        u_hat_.add(one, q_, one);
        if (this->precond_ != nullptr) {
            this->precond_->apply(&u_hat_);
        }

        soln->add_scale(alpha, u_hat_);
        mat.multiply(u_hat_, &q_hat_);
        r_.add_scale(-alpha, q_hat_);

        this->res_norm_ = r_.norm();
        //printf("i: %d r: %e \n", this->it_counter_,  this->res_norm_);
        if (this->is_converged_()) {
            break;
        }

        // prepare for next iteration
        beta = one/rho;
        rho = r_hat_.dot(r_);
        beta = beta * rho;
    }
    clear();
}

// instantiate template classes
template class CGS<HostMatrix<double>, HostVector<double>, double>;
template class CGS<HostMatrix<float>, HostVector<float>, float>;

#ifdef __CUDACC__
template class CGS<DeviceMatrix<double>, DeviceVector<double>, double>;
template class CGS<DeviceMatrix<float>, DeviceVector<float>, float>;
#endif
