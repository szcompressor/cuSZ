/**
 * @file binding.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-10-06
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_BINDING_HH
#define CUSZ_BINDING_HH

// NVCC requires the two headers; clang++ does not.
#include <limits>
#include <type_traits>

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

namespace cusz {

/**
 * ------------
 * default path
 * ------------
 *
 * Predictor<T, E, (FP)>
 *           |  |   ^
 *           v  |   |
 *   SpCodec<T> |   +---- default "fast-lowlowprecision"
 *              v
 *      Encoder<E, H>
 */

template <class Predictor, class SpCodec, class Codec, class FallbackCodec>
struct CompressorTemplate {
    using T1 = typename Predictor::Origin;
    using T2 = typename Predictor::Anchor;
    using E1 = typename Predictor::ErrCtrl;
    using T3 = typename SpCodec::Origin;  // SpCodec -> BYTE, omit
    using E2 = typename Codec::Origin;
    using H  = typename Codec::Encoded;
    // fallback
    using E3   = typename FallbackCodec::Origin;
    using H_FB = typename FallbackCodec::Encoded;

    using PREDICTOR      = Predictor;
    using SPCODEC        = SpCodec;
    using CODEC          = Codec;
    using FALLBACK_CODEC = FallbackCodec;

    static void type_matching()
    {
        static_assert(
            std::is_same<T1, T2>::value and std::is_same<T1, T3>::value,
            "Predictor::Origin, Predictor::Anchor, and SpCodec::Origin must be the same.");

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

    template <class Stage1, class Stage2>
    static size_t get_len_uncompressed(Stage1* s, Stage2*)
    {
        // !! The compiler does not support/generate constexpr properly
        // !! just put combinations
        if CONSTEXPR (std::is_same<Stage1, Predictor>::value and std::is_same<Stage2, SpCodec>::value)
            return s->get_len_outlier();

        if CONSTEXPR (std::is_same<Stage1, Predictor>::value and std::is_same<Stage2, Codec>::value)  //
            return s->get_len_quant();
    }
};

/**
 * -------------
 * sp-aware path
 * -------------
 *
 * Predictor<T, E, (FP)>
 *              |
 *              v
 *      SpCodec<E>
 */

/**
 * @deprecated There will be only one compression pipeline, featuring auto-switching between "default" and
 * sparsity-aware path.
 */
template <class Predictor, class SpCodec>
struct PredictorReducerBinding {
    using T1 = typename Predictor::Origin;
    using T2 = typename Predictor::Anchor;
    using E1 = typename Predictor::ErrCtrl;
    using E2 = typename SpCodec::Origin;

    using PREDICTOR = Predictor;
    using SPCODEC   = SpCodec;

    // SpRecuder -> BYTE, omit

    static void type_matching()
    {
        static_assert(std::is_same<T1, T2>::value, "Predictor::Origin and Predictor::Anchor must be the same.");

        // alternatively, change Output of Predictor in place of Origin
        static_assert(std::is_same<E1, E2>::value, "Predictor::ErrCtrl and SpCodec::Origin must be the same.");

        // TODO this is the restriction for now.
        static_assert(std::is_floating_point<T1>::value, "Predictor::Origin must be floating-point type.");

        // TODO this is the restriction for now.
        static_assert(std::is_floating_point<E1>::value, "Predictor::ErrCtrl must be floating-point type.");
    }

    template <class Context>
    static size_t get_spcodec_input_len(Context* ctx)
    {
        return ctx->quant_len;
    }
};

}  // namespace cusz

#endif
