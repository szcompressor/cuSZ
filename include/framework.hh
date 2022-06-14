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

#include "component.hh"
#include "compressor.hh"

namespace cusz {

template <typename InputDataType>
struct PredefinedCombination {
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
    using ERRCTRL = ErrCtrlTrait<4, false>::type;       // predefined
    using FP      = FastLowPrecisionTrait<true>::type;  // predefined
    using Huff4   = HuffTrait<4>::type;
    using Huff8   = HuffTrait<8>::type;
    using Meta4   = MetadataTrait<4>::type;

    template <class Predictor, class Spcodec, class Codec, class FallbackCodec>
    struct CompressorTemplate;

    /* Predictor */
    using PredictionUnified = typename cusz::PredictionUnified<DATA, ERRCTRL, FP>;
    // using PredictorSpline3 = typename cusz::PredictorSpline3<DATA, ERRCTRL, FP>;

    /* Lossless Spcodec */
    using SpcodecMat = typename cusz::SpcodecCSR<DATA, Meta4>;
    using SpcodecVec = typename cusz::SpcodecVec<DATA, Meta4>;

    /* Lossless Codec*/
    using CodecHuffman32 = cusz::HuffmanCoarse<ERRCTRL, Huff4, Meta4>;
    using CodecHuffman64 = cusz::HuffmanCoarse<ERRCTRL, Huff8, Meta4>;

    /* Predefined Combination */
    using LorenzoFeatured = CompressorTemplate<PredictionUnified, SpcodecVec, CodecHuffman32, CodecHuffman64>;
    // using LorenzoFeatured = CompressorTemplate<PredictionUnified, SpcodecMat, CodecHuffman32, CodecHuffman64>;
    // using Spline3Featured = CompressorTemplate<PredictorSpline3, SpcodecVec, CodecHuffman32, CodecHuffman64>;
};

template <typename InputDataType>
template <class PREDICTOR, class SPCODEC, class CODEC, class FALLBACK_CODEC>
struct PredefinedCombination<InputDataType>::CompressorTemplate {
   public:
    using Predictor     = PREDICTOR;
    using Spcodec       = SPCODEC;
    using Codec         = CODEC;
    using FallbackCodec = FALLBACK_CODEC;

   private:
    using T1 = typename Predictor::Origin;
    using T2 = typename Predictor::Anchor;
    using E1 = typename Predictor::ErrCtrl;
    using T3 = typename Spcodec::Origin;  // Spcodec -> BYTE, omit
    using E2 = typename Codec::Origin;
    using H  = typename Codec::Encoded;
    // fallback
    using E3   = typename FallbackCodec::Origin;
    using H_FB = typename FallbackCodec::Encoded;

    static void type_matching()
    {
        static_assert(
            std::is_same<T1, T2>::value and std::is_same<T1, T3>::value,
            "Predictor::Origin, Predictor::Anchor, and Spcodec::Origin must be the same.");

        static_assert(std::is_same<E1, E2>::value, "Predictor::ErrCtrl and Codec::Origin must be the same.");
        static_assert(std::is_same<E1, E3>::value, "Predictor::ErrCtrl and FallbackCodec::Origin must be the same.");

        // TODO this is the restriction for now.
        static_assert(std::is_floating_point<T1>::value, "Predictor::Origin must be floating-point type.");

        // TODO open up the possibility of (E1 neq E2) and (E1 being FP)
        static_assert(
            std::numeric_limits<E1>::is_integer and std::is_unsigned<E1>::value,
            "Predictor::ErrCtrl must be unsigned integer.");

        static_assert(
            std::numeric_limits<H>::is_integer and std::is_unsigned<H>::value,
            "Codec::Encoded must be unsigned integer.");

        // fallback
        static_assert(
            std::numeric_limits<H_FB>::is_integer and std::is_unsigned<H>::value,
            "Codec::Encoded must be unsigned integer.");
    }
};

template <typename DATA = float>
struct Framework {
    /* Usable Compressor */
    using DefaultCompressor         = class Compressor<typename PredefinedCombination<DATA>::LorenzoFeatured>;
    using LorenzoFeaturedCompressor = class Compressor<typename PredefinedCombination<DATA>::LorenzoFeatured>;
    // in progress
    // using Spline3FeaturedCompressor = class Compressor<typename PredefinedCombination<DATA>::Spline3Featured>;
};

}  // namespace cusz

#endif
