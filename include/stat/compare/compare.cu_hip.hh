#ifndef C81325C8_34E8_443A_AC64_8524112A367F
#define C81325C8_34E8_443A_AC64_8524112A367F

#include "busyheader.hh"

namespace psz {

namespace cu_hip {

template <typename T>
void extrema(T* d_ptr, size_t len, T res[4]);

}

template <typename T>
bool cuda_error_bounded(
    T* a, T* b, size_t const len, double const eb,
    size_t* first_faulty_idx = nullptr);

}  // namespace psz

#endif /* C81325C8_34E8_443A_AC64_8524112A367F */
