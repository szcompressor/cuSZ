#ifndef B6162392_EE38_4B7F_AE43_C891E283B221
#define B6162392_EE38_4B7F_AE43_C891E283B221

#include "cusz/type.h"

namespace psz {

template <typename T>
void dpl_get_extrema_rawptr(T* d_ptr, size_t len, T res[4]);

bool dpl_identical(void* d1, void* d2, size_t sizeof_T, size_t const len);

template <typename T>
bool dpl_error_bounded(
    T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx);

template <typename T>
void dpl_get_maxerr(
    T* reconstructed,     // in
    T* original,          // in
    size_t len,           // in
    T& maximum_val,       // out
    size_t& maximum_loc,  // out
    bool destructive = false);

template <typename T>
void dpl_assess_quality(psz_summary* s, T* xdata, T* odata, size_t const len);

}  // namespace psz

#endif /* B6162392_EE38_4B7F_AE43_C891E283B221 */
