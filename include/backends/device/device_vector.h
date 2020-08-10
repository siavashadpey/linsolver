#ifndef DEVICE_VECTOR_H
#define DEVICE_VECTOR_H

#include <cublas_v2.h>

#include "backends/base_vector.h"
#include "backends/device/device_matrix.h"

/**
 * \brief Implementation of a Vector class on the NVIDIA GPU system, referred to as the device system.
 */
template <typename NumType>
class DeviceVector: public BaseVector<NumType> {
public:
    /**
     * Default constructor.
     */
    DeviceVector();

    /**
     * Destructor.
     */
    virtual ~DeviceVector();

    virtual void allocate(int n);
    virtual void clear();

    /**
     * Copy from the inputted data to the vector.
     * @param[in] w The pointer to the data (on the device system) to copy from.
     * @note allocate should be called first.
     */
    virtual void copy_from(const NumType* w);
    virtual void copy_from(const BaseVector<NumType>& w);

    /**
     * Copy from the inputted data to the vector.
     * @param[in] w The pointer to the data (on the host system) to copy from.
     * @note allocate should be called first.
     */
    virtual void copy_from_host(const NumType* w);

    /**
     * Output the vector's values.
     * @param[out] w The pointer to the data (on the device system) to copy to.
     * @note device memory should be allocated to \p w before calling this function.
     */
    virtual void copy_to(NumType* w) const;
    virtual void copy_to(BaseVector<NumType>& w) const;

    /**
     * Output the vector's values.
     * @param[out] w The pointer to the data (on the host system) to copy to.
     * @note device memory should be allocated to \p w before calling this function.
     */
    virtual void copy_to_host(NumType* w) const;

    virtual NumType norm() const;
    virtual NumType dot(const BaseVector<NumType>& w) const;
    virtual void zeros();
    virtual void ones();
    virtual void scale(NumType alpha);
    virtual void add(NumType alpha, const BaseVector<NumType>& w, NumType beta);
    virtual void add_scale(NumType alpha, const BaseVector<NumType>& w);
    virtual void scale_add(NumType alpha, const BaseVector<NumType>& w);

protected:
    /**
     * Values of the vector.
     */
    NumType* vec_;

    /**
     * Cublas handle.
     */
    cublasHandle_t cublasHandle_;

private:
    // befriending classes
    friend class DeviceMatrix<NumType>; 
};
#endif
