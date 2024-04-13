/**
 * @file compare.thrust.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-09
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef B0EE0E82_B3AA_4946_A589_A3A6A83DD862
#define B0EE0E82_B3AA_4946_A589_A3A6A83DD862

#include "cusz/type.h"

namespace psz {

template <typename T>
void thrustgpu_get_extrema_rawptr(T* d_ptr, size_t len, T res[4]);

bool thrustgpu_identical(
    void* d1, void* d2, size_t sizeof_T, size_t const len);

template <typename T>
bool thrustgpu_error_bounded(
    T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx);

template <typename T>
void thrustgpu_get_maxerr(
    T* reconstructed,     // in
    T* original,          // in
    size_t len,           // in
    T& maximum_val,       // out
    size_t& maximum_loc,  // out
    bool destructive = false);

template <typename T>
void thrustgpu_assess_quality(
    psz_summary* s, T* xdata, T* odata, size_t const len);

}  // namespace psz

#endif /* B0EE0E82_B3AA_4946_A589_A3A6A83DD862 */
