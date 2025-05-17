#ifndef PSZ_DETAIL_TYPING_HH
#define PSZ_DETAIL_TYPING_HH

#include "cusz/type.h"
#include "cxx_typing.h"

/**
 * @brief CUDA API does not accept u8 (understandable by literal), but instead,
 * `unsigned long long`, which is ambiguous anyway.
 */
template <typename T>
using cuszCOMPAT = _portable::CudaCompat<T>;

template <bool LARGE>
using LargeInputTrait = _portable::LargeInputTrait<LARGE>;

template <bool FAST>
using FastLowPrecisionTrait = _portable::FastLowPrecisionTrait<FAST>;

template <psz_dtype T>
using Ctype = _portable::Ctype<T>;

template <typename Ctype>
using PszType = _portable::TypeSym<Ctype>;

#endif /* PSZ_DETAIL_TYPING_HH */
