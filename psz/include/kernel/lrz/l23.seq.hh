#ifndef A976A9C2_ACC0_4F3E_9840_E1ABB3AE6E82
#define A976A9C2_ACC0_4F3E_9840_E1ABB3AE6E82

#include "cusz/nd.h"
#include "cusz/type.h"
#include "mem/cxx_sp_cpu.h"

template <typename T, typename Eq>
pszerror CPU_c_lorenzo_nd_with_outlier(
    T* const in_data, psz_dim3 const data_len3, Eq* const out_eq, void* out_outlier, f8 const eb,
    uint16_t const radius, float* time_elapsed);

template <typename T, typename Eq>
pszerror CPU_x_lorenzo_nd(
    Eq* const in_eq, T* const in_outlier, T* const out_data, psz_dim3 const data_len3, f8 const eb,
    uint16_t const radius, f4* time_elapsed);

#endif /* A976A9C2_ACC0_4F3E_9840_E1ABB3AE6E82 */
