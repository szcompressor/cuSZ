/**
 * @file default_path.cuh
 * @author Jiannan Tian
 * @brief cuSZ compressor of the default path
 * @version 0.3
 * @date 2021-10-05
 * (create) 2020-02-12; (release) 2020-09-20;
 * (rev.1) 2021-01-16; (rev.2) 2021-07-12; (rev.3) 2021-09-06; (rev.4) 2021-10-05
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_DEFAULT_PATH_CUH
#define CUSZ_DEFAULT_PATH_CUH

#include "base_cusz.cuh"
#include "binding.hh"
#include "wrapper.hh"

template <class BINDING>
class DefaultPathCompressor : public BaseCompressor<typename BINDING::PREDICTOR> {
   public:
    using Predictor = typename BINDING::PREDICTOR;
    using SpReducer = typename BINDING::SPREDUCER;
    using Encoder   = typename BINDING::ENCODER;

    using BYTE = uint8_t;
    using T    = typename Predictor::Origin;
    using FP   = typename Predictor::Precision;
    using E    = typename Predictor::ErrCtrl;
    using H    = typename Encoder::Encoded;

   private:
    // --------------------
    // not in base class
    // --------------------
    Capsule<H>      book;
    Capsule<H>      huff_data;
    Capsule<size_t> huff_counts;
    Capsule<BYTE>   revbook;
    Capsule<BYTE>   sp_use;

    Predictor* predictor;
    SpReducer* csr;
    Encoder*   codec;

    size_t   m, mxm;
    uint32_t sp_dump_nbyte;

   private:
    uint32_t tune_deflate_chunksize(size_t len);
    // TODO better move to base compressor
    DefaultPathCompressor& analyze_compressibility();
    DefaultPathCompressor& internal_eval_try_export_book();
    DefaultPathCompressor& internal_eval_try_export_quant();
    DefaultPathCompressor& try_skip_huffman();
    DefaultPathCompressor& get_freq_codebook();
    DefaultPathCompressor& huffman_encode();

   public:
    uint32_t get_decompress_space_len() { return mxm + ChunkingTrait<1>::BLOCK; }

   public:
    DefaultPathCompressor(cuszCTX* _ctx, Capsule<T>* _in_data);
    DefaultPathCompressor(cuszCTX* _ctx, Capsule<BYTE>* _in_dump);
    ~DefaultPathCompressor();

    DefaultPathCompressor& compress();

    template <cuszLOC SRC, cuszLOC DST>
    DefaultPathCompressor& consolidate(BYTE** dump);

    DefaultPathCompressor& decompress(Capsule<T>* out_xdata);
    DefaultPathCompressor& backmatter(Capsule<T>* out_xdata);
};

struct DefaultPath {
    using DATA    = DataTrait<4>::type;
    using ERRCTRL = ErrCtrlTrait<2>::type;
    using FP      = FastLowPrecisionTrait<true>::type;

    using DefaultBinding = PredictorReducerEncoderBinding<
        cusz::PredictorLorenzo<DATA, ERRCTRL, FP>,
        cusz::CSR10<DATA>,
        cusz::HuffmanWork<ERRCTRL, HuffTrait<4>::type>>;

    using DefaultCompressor = class DefaultPathCompressor<DefaultBinding>;

    using FallbackBinding = PredictorReducerEncoderBinding<
        cusz::PredictorLorenzo<DATA, ERRCTRL, FP>,
        cusz::CSR10<DATA>,
        cusz::HuffmanWork<ERRCTRL, HuffTrait<8>::type>>;

    using FallbackCompressor = class DefaultPathCompressor<FallbackBinding>;
};

#endif