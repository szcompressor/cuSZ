/**
 * @file tehm.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 * (create) 2021-10-06 (rev) 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_FRAMEWORK
#define CUSZ_FRAMEWORK

#include "compressor.hh"
#include "cusz/type.h"
#include "hf/hf.hh"

namespace cusz {

using TimeRecordTuple = std::tuple<const char*, double>;
using TimeRecord = std::vector<TimeRecordTuple>;
using timerecord_t = TimeRecord*;

};  // namespace cusz

namespace cusz {

template <typename InDtype, bool FastLowPrecision = true>
struct TEHM {
 public:
  /**
   *
   *  Compressor<T, E, (FP)>
   *             |  |   ^
   *  outlier <--+  |   +---- default fast-low-precision
   *                v
   *        Encoder<E, H>
   */

  using T = InDtype;
  using E = ErrCtrlTrait<4, false>::type;  // predefined for mem. overlapping
  using FP = typename FastLowPrecisionTrait<FastLowPrecision>::type;
  using H = HuffTrait<4>::type;
  using H8 = HuffTrait<4>::type;
  using M = MetadataTrait<4>::type;

  /* Lossless Codec*/
  using CodecHuffman32 = cusz::HuffmanCodec<E, H, M>;
  using CodecHuffman64 = cusz::HuffmanCodec<E, H8, M>;
  using Codec = CodecHuffman32;
  using FallbackCodec = CodecHuffman64;
};

using CompressorF4 = cusz::Compressor<cusz::TEHM<f4>>;
using CompressorF8 = cusz::Compressor<cusz::TEHM<f8>>;

}  // namespace cusz

#endif
