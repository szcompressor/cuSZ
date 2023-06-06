/**
 * @file framework.hh
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
#include "hf/hf.hh"
/* removed, leaving vacancy in framework description
 * #include "pipeline/prediction.inl"
 */
#include "pipeline/spcodec.inl"

namespace cusz {

template <typename InputDataType, bool FastLowPrecision = true>
struct Framework {
   public:
    /**
     *
     *   Predictor<T, E, (FP)>
     *             |  |   ^
     *             v  |   |
     *     Spcodec<T> |   +---- default "fast-lowlowprecision"
     *                v
     *        Encoder<E, H>
     */

    using DATA    = InputDataType;
    using ERRCTRL = ErrCtrlTrait<4, false>::type;  // predefined for mem. overlapping
    using FP      = typename FastLowPrecisionTrait<FastLowPrecision>::type;
    using Huff4   = HuffTrait<4>::type;
    using Huff8   = HuffTrait<8>::type;
    using Meta4   = MetadataTrait<4>::type;

    template <class Codec, class FallbackCodec>
    struct CompressorTemplate;

    /* Predictor vacancy */

    using CompatPurposeSpcodec = typename cusz::SpcodecVec<DATA, Meta4>;
    using Spcodec              = CompatPurposeSpcodec;

    /* Lossless Codec*/
    using CodecHuffman32 = cusz::HuffmanCodec<ERRCTRL, Huff4, Meta4>;
    using CodecHuffman64 = cusz::HuffmanCodec<ERRCTRL, Huff8, Meta4>;
    using Codec          = CodecHuffman32;
    using FallbackCodec  = CodecHuffman64;
};

using CompressorFP32 = cusz::Compressor<cusz::Framework<float>>;

}  // namespace cusz

#endif
