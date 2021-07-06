/**
 * @file thrust_hist.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.4
 * @date 2020-11-28
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 *
 */

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace par_ops {
namespace use_thrust {

template <typename T>
void Histogram(
    T*            d_data,
    unsigned int  len,
    unsigned int* d_keys,
    unsigned int* d_hist,
    unsigned int  nbin,
    T*            min_value = nullptr,
    T*            max_value = nullptr);

}
}  // namespace par_ops

template <typename T>
void par_ops::use_thrust::Histogram(
    T*            d_data,
    unsigned int  len,
    unsigned int* d_keys,
    unsigned int* d_hist,
    unsigned int  nbin,
    T*            min_value,
    T*            max_value)
{
    thrust::device_ptr<T> p_data = thrust::device_pointer_cast(d_data);
    // unsorted
    thrust::device_ptr<unsigned int> p_keys = thrust::device_pointer_cast(d_keys);
    thrust::device_ptr<unsigned int> p_hist = thrust::device_pointer_cast(d_hist);

    if (not(min_value and max_value)) {
        auto min_el = thrust::min_element(p_data, p_data + len) - p_data;
        auto max_el = thrust::max_element(p_data, p_data + len) - p_data;
        *min_value  = *(p_data + min_el);
        *max_value  = *(p_data + max_el);
    }
    double rng  = *max_value - *min_value;
    double step = rng / nbin;

    thrust::device_reference<T> ref_to_min_value =  *min_value;


    auto gen_key = [&] __device__( T val) { return static_cast<unsigned int>(ceil((val - ref_to_min_value) / step)); };

    thrust::equal_to<unsigned int> binary_pred;
    thrust::plus<unsigned int>     binary_op;

    //    thrust::pair<unsigned int*, unsigned int*> new_end =
    thrust::reduce_by_key(
        /* keys first   */ thrust::make_transform_iterator(p_data, gen_key),
        /* keys last    */ thrust::make_transform_iterator(p_data + len, gen_key),
        /* values first */ thrust::make_constant_iterator(1),
        /* keys out     */ p_keys,
        /* values out   */ p_hist,
        /**/ binary_pred,
        /**/ binary_op);

    // sort by key
}