/**
 * @file sp_path.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-29
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_SP_PATH_CUH
#define CUSZ_SP_PATH_CUH

#include "base_cusz.cuh"
#include "binding.hh"
#include "wrapper.hh"
#include "wrapper/interp_spline3.cuh"

template <class BINDING>
class SpPathCompressor : public BaseCompressor<typename BINDING::PREDICTOR> {
   public:
    using Predictor = typename BINDING::PREDICTOR;
    using SpReducer = typename BINDING::SPREDUCER;

    using BYTE = uint8_t;
    using T    = typename Predictor::Origin;
    using FP   = typename Predictor::Precision;
    using E    = typename Predictor::ErrCtrl;

   private:
    // --------------------
    // not in base class
    // --------------------
    Capsule<BYTE> sp_use;

    Predictor* predictor;
    SpReducer* spreducer;

    size_t   m, mxm;
    uint32_t sp_dump_nbyte;

    static const auto EXEC_SPACE     = cuszLOC::DEVICE;
    static const auto FALLBACK_SPACE = cuszLOC::HOST;
    static const auto BOTH           = cuszLOC::HOST_DEVICE;

   private:
   public:
    SpPathCompressor(cuszCTX* _ctx, Capsule<T>* _in_data);
    SpPathCompressor(cuszCTX* _ctx, Capsule<BYTE>* _in_dump);
    ~SpPathCompressor();

    SpPathCompressor& compress();

    template <cuszLOC SRC, cuszLOC DST>
    SpPathCompressor& consolidate(BYTE** dump);

    SpPathCompressor& decompress(Capsule<T>* out_xdata);
    SpPathCompressor& backmatter(Capsule<T>* out_xdata);
};

struct SparsityAwarePath {
    using DATA    = DataTrait<4>::type;
    using ERRCTRL = ErrCtrlTrait<4, true>::type;
    using FP      = FastLowPrecisionTrait<true>::type;

    using DefaultBinding = PredictorReducerBinding<  //
        cusz::Spline3<DATA, ERRCTRL, FP>,
        cusz::CSR11<ERRCTRL>>;

    using DefaultCompressor = class SpPathCompressor<DefaultBinding>;

    using FallbackBinding = PredictorReducerBinding<  //
        cusz::PredictorLorenzo<DATA, ERRCTRL, FP>,
        cusz::CSR11<ERRCTRL>>;

    using FallbackCompressor = class SpPathCompressor<FallbackBinding>;
};

#endif
