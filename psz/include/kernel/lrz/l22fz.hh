#ifndef F433ECF5_7709_4A09_A500_EE58F1B7FF64
#define F433ECF5_7709_4A09_A500_EE58F1B7FF64

#include <cstdint>

#include "cusz/type.h"
#include "mem/cxx_sp_gpu.h"
#include "port.hh"

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)

namespace psz::cuhip {

using FzgpuDeltaType = uint16_t;

template <typename T, typename Eq>
pszerror GPU_c_lorenzo_nd_with_FZGPU(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    void* out_outlier, PROPER_EB const eb, uint16_t const radius,
    f4* time_elapsed, void* stream);

template <typename T>
pszerror GPU_c_lorenzo_nd_FZGPU_delta_only(
    T* data, DeltaT* delta, bool* signum, dim3 const len3, double const eb,
    float* time_elapsed, cudaStream_t stream);

template <typename T, typename Eq>
pszerror GPU_x_lorenzo_nd_FZGPU(
    Eq* in_eq, T* in_outlier, T* out_data, dim3 const data_len3,
    PROPER_EB const eb, uint16_t const radius, f4* time_elapsed, void* stream);

}  // namespace psz::cuhip

#endif

#endif /* F433ECF5_7709_4A09_A500_EE58F1B7FF64 */
