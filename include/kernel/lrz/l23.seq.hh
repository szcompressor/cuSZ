#ifndef A976A9C2_ACC0_4F3E_9840_E1ABB3AE6E82
#define A976A9C2_ACC0_4F3E_9840_E1ABB3AE6E82

#include "cusz/nd.h"
#include "cusz/type.h"
#include "mem/compact/compact.seq.hh"

template <
    typename T, typename EQ, typename FP = T,
    typename OUTLIER = CompactSerial<T>>
pszerror psz_comp_l23_seq(
    T* const data, psz_dim3 const len3, double const eb, int const radius,
    EQ* const eq, OUTLIER* outlier, float* time_elapsed);

template <typename T, typename EQ, typename FP = T>
pszerror psz_decomp_l23_seq(
    EQ* eq, psz_dim3 const len3, T* outlier, f8 const eb, int const radius,
    T* xdata, f4* time_elapsed);

#endif /* A976A9C2_ACC0_4F3E_9840_E1ABB3AE6E82 */
