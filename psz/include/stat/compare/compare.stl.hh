/**
 * @file compare.stl.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-09
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef C93C3857_8821_4988_B6F0_4E885060F642
#define C93C3857_8821_4988_B6F0_4E885060F642

#include "cusz/type.h"

namespace psz {

template <typename T>
bool cppstl_identical(T* d1, T* d2, size_t const len);

template <typename T>
void cppstl_extrema(T* in, szt const len, T res[4]);

template <typename T>
bool cppstl_error_bounded(
    T* a, T* b, szt const len, f8 const eb, szt* first_faulty_idx = nullptr);

template <typename T>
void cppstl_assess_quality(pszsummary* s, T* xdata, T* odata, szt const len);

}  // namespace psz

#endif /* C93C3857_8821_4988_B6F0_4E885060F642 */
