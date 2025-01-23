#ifndef PSZ_MODULE_LRZ_SEQ_HH
#define PSZ_MODULE_LRZ_SEQ_HH

#include "cusz/nd.h"
#include "cusz/type.h"

template <typename T, typename Eq>
pszerror CPU_c_lorenzo_nd_with_outlier(
    T* const in_data, psz_dim3 const data_len3, Eq* const out_eq, void* out_outlier, f8 const eb,
    uint16_t const radius, float* time_elapsed);

template <typename T, typename Eq>
pszerror CPU_x_lorenzo_nd(
    Eq* const in_eq, T* const in_outlier, T* const out_data, psz_dim3 const data_len3, f8 const eb,
    uint16_t const radius, f4* time_elapsed);

#endif /* PSZ_MODULE_LRZ_SEQ_HH */
