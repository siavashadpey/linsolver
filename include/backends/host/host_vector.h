#ifndef HOST_VECTOR_H
#define HOST_VECTOR_H

#include "backends/base_vector.h"
#include "backends/host/host_matrix.h"

#ifdef __CUDACC__
#include "backends/device/device_vector.h"
#endif

/**
 * \brief Implementation of a Vector class on the host system.
 * \tparam NumType Number type (double and float currently supported).
 */
template <typename NumType>
class HostVector: public BaseVector<NumType> {
public:
    /**
     * Default constructor.
     */
    HostVector();

    /**
     * Destructor.
     */
    virtual ~HostVector();

    virtual void allocate(int n);
    virtual void clear();
    virtual void copy_from(const NumType* w);
    virtual void copy_from(const BaseVector<NumType>& w);
    virtual void copy_to(NumType* w) const;
    virtual void copy_to(BaseVector<NumType>& w) const;
    virtual NumType& operator[](int i);
    virtual NumType norm() const;
    virtual NumType dot(const BaseVector<NumType>& w) const;
    virtual void zeros();
    virtual void ones();
    virtual void scale(NumType alpha);
    virtual void add(NumType alpha, const BaseVector<NumType>& w, NumType beta);
    virtual void add_scale(NumType alpha, const BaseVector<NumType>& w);
    virtual void scale_add(NumType alpha, const BaseVector<NumType>& w);
    virtual void pointwise_multiply(const BaseVector<NumType>& w);

protected:
    /**
     * Values of the vector.
     */
    NumType* vec_;

private:
    // befriending classes
    friend class HostMatrix<NumType>;
    friend class HostMatrixCOO<NumType>;

#ifdef __CUDACC__
    friend class DeviceVector<NumType>;
#endif
};
#endif
