#ifndef ILU_H
#define ILU_H

#include "solvers/base_preconditioner.h"
/**
 * \brief Incomplete LU (ILU) factorization preconditoner based on fill-in levels.
 * \tparam MatType Matrix type (HostMatrix<NumType> and DeviceMatrix<NumTyper> currently supported).
 * \tparam VecType Vector type (HostVector<NumType> and DeviceVector<NumTyper> currently supported).
 * \tparam NumType Number type (double and float currently supported).
 * @note The SSOR relaxation parameter is set to one.
 */
template <class MatType, class VecType, typename NumType>
class ILU: public BasePreconditioner<MatType, VecType, NumType> {
public:
    ILU();
    ~ILU();

    /**
     * @param[in] level The fill-in level to be used by the ILU preconditioner.
     */
    void set_level(int level);

    /**
     * @param[in] tau The threshold tolerance used for ILUT.
     * @note The fill-in level must be one or greater for ILUT.
     */
    void set_threshold(int tau);

    void prepare_preconditioner(const MatType& mat);
    void apply(VecType* x) const;
    virtual void apply(const VecType& b, VecType* x) const;

protected:
    int p_;
    NumType tau_;
    MatType ilu_mat_;
};
#endif