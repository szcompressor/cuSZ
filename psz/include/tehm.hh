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
#include "hfclass.hh"

namespace psz {

using TimeRecordTuple = std::tuple<const char*, double>;
using TimeRecord = std::vector<TimeRecordTuple>;
using timerecord_t = TimeRecord*;

};  // namespace psz

namespace cusz {

template <typename Input, bool Fast = true>
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

  using T = Input;
  using E = uint16_t;
  using FP = typename FastLowPrecisionTrait<Fast>::type;
  using M = uint32_t;

  /* Lossless Codec*/
  using Codec = cusz::HuffmanCodec<E, M>;
};

using CompressorF4 = cusz::Compressor<cusz::TEHM<f4>>;
using CompressorF8 = cusz::Compressor<cusz::TEHM<f8>>;

}  // namespace cusz

#endif
