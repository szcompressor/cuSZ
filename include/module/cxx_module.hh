
#ifndef E85FA9DC_9350_4F91_B4D8_5C094661479E
#define E85FA9DC_9350_4F91_B4D8_5C094661479E

#include "cusz/cxx_array.hh"

namespace _2401 {

template <typename T>
pszerror pszcxx_predict_lorenzo(
    pszarray_cxx<T> in, pszrc2 const rc, pszarray_cxx<u4> out_errquant,
    pszcompact_cxx<T> out_outlier, void* stream);

template <typename T>
pszerror pszcxx_reverse_predict_lorenzo(
    pszarray_cxx<u4> in_errquant, pszarray_cxx<T> in_scattered_outlier,
    pszrc2 const rc, pszarray_cxx<T> out_reconstruct, void* stream);

};  // namespace _2401

#endif /* E85FA9DC_9350_4F91_B4D8_5C094661479E */
