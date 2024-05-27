/**
 * @file _compare.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-08
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef C9DB8B27_F5D7_454F_9485_CAD4B2FE4A92
#define C9DB8B27_F5D7_454F_9485_CAD4B2FE4A92

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "stat/compare/compare.dpl.hh"
#include <dpct/dpl_utils.hpp>

// #include <thrust/device_vector.h>

namespace psz {

static const int MINVAL = 0;
static const int MAXVAL = 1;
static const int AVGVAL = 2;
static const int RNG    = 3;

template <typename T>
void thrustgpu_get_extrema_rawptr(T* d_ptr, size_t len, T res[4])
{
    dpct::device_pointer<T> g_ptr = dpct::get_device_pointer(d_ptr);

    auto minel =
        std::min_element(oneapi::dpl::execution::seq, g_ptr, g_ptr + len) -
        g_ptr;
    auto maxel =
        std::max_element(oneapi::dpl::execution::seq, g_ptr, g_ptr + len) -
        g_ptr;
    res[MINVAL] = *(g_ptr + minel);
    res[MAXVAL] = *(g_ptr + maxel);
    res[RNG]    = res[MAXVAL] - res[MINVAL];

    auto sum = std::reduce(
        oneapi::dpl::execution::seq, g_ptr, g_ptr + len, (T)0.0,
        std::plus<T>());
    res[AVGVAL] = sum / len;
}

// commented for better build time
/*
template <typename T>
void thrustgpu_get_extrema(thrust::device_ptr<T> g_ptr, size_t len, T res[4])
{
    auto minel  = thrust::min_element(g_ptr, g_ptr + len) - g_ptr;
    auto maxel  = thrust::max_element(g_ptr, g_ptr + len) - g_ptr;
    res[MINVAL] = *(g_ptr + minel);
    res[MAXVAL] = *(g_ptr + maxel);
    res[RNG]    = res[MAXVAL] - res[MINVAL];

    auto sum    = thrust::reduce(g_ptr, g_ptr + len, (T)0.0, thrust::plus<T>());
    res[AVGVAL] = sum / len;
}
*/

}  // namespace psz

#endif /* C9DB8B27_F5D7_454F_9485_CAD4B2FE4A92 */
