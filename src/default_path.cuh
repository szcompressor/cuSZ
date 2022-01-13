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

#include "base_compressor.cuh"
#include "binding.hh"
#include "wrapper.hh"
#include "wrapper/spgs.cuh"

template <class BINDING>
class DefaultPathCompressor : public BaseCompressor<typename BINDING::PREDICTOR> {
   public:
    using Predictor = typename BINDING::PREDICTOR;
    using SpReducer = typename BINDING::SPREDUCER;
    using Codec     = typename BINDING::CODEC;

    using BYTE = uint8_t;
    using T    = typename Predictor::Origin;
    using FP   = typename Predictor::Precision;
    using E    = typename Predictor::ErrCtrl;
    using H    = typename Codec::Encoded;
    using M    = typename Codec::MetadataT;

   private:
   private:
    // --------------------
    // not in base class
    // --------------------
    Capsule<H>    book;
    Capsule<H>    huff_data;
    Capsule<M>    huff_counts;
    Capsule<BYTE> revbook;
    Capsule<BYTE> sp_use;

    // tmp, device only
    Capsule<int> ext_rowptr;
    Capsule<int> ext_colidx;
    Capsule<T>   ext_values;

    H* huff_workspace;  // compress

    struct {
        Capsule<H>    in;
        Capsule<M>    meta;
        Capsule<BYTE> revbook;
    } xhuff;

    Predictor* predictor;
    SpReducer* spreducer;
    Codec*     codec;

    size_t   m, mxm;
    uint32_t sp_dump_nbyte;

   private:
    // TODO better move to base compressor
    DefaultPathCompressor& analyze_compressibility();
    DefaultPathCompressor& internal_eval_try_export_book();
    DefaultPathCompressor& internal_eval_try_export_quant();
    DefaultPathCompressor& try_skip_huffman();
    // DefaultPathCompressor& get_freq_codebook();
    // DefaultPathCompressor& old_huffman_encode();

   public:
    DefaultPathCompressor(cuszCTX* _ctx, Capsule<T>* _in_data, uint3 xyz, int dict_size);
    DefaultPathCompressor(cuszCTX* _ctx, Capsule<BYTE>* _in_dump);
    ~DefaultPathCompressor();

    DefaultPathCompressor& compress(bool optional_release_input = false);

    template <cusz::LOC SRC, cusz::LOC DST>
    DefaultPathCompressor& consolidate(BYTE** dump);

    DefaultPathCompressor& decompress(Capsule<T>* out_xdata);
    DefaultPathCompressor& backmatter(Capsule<T>* out_xdata);
};

struct DefaultPath {
    using DATA    = DataTrait<4>::type;
    using ERRCTRL = ErrCtrlTrait<2>::type;
    using FP      = FastLowPrecisionTrait<true>::type;

    using DefaultBinding = PredictorReducerCodecBinding<
        cusz::PredictorLorenzo<DATA, ERRCTRL, FP>,
        cusz::CSR11<DATA>,
        // cusz::spGS<DATA>,  //  not woking for CUDA 10.2 on ppc
        cusz::HuffmanCoarse<ERRCTRL, HuffTrait<4>::type, MetadataTrait<4>::type>  //
        >;

    using DefaultCompressor = class DefaultPathCompressor<DefaultBinding>;

    using FallbackBinding = PredictorReducerCodecBinding<
        cusz::PredictorLorenzo<DATA, ERRCTRL, FP>,
        cusz::CSR11<DATA>,
        // cusz::spGS<DATA>,  //  not woking for CUDA 10.2 ppc
        cusz::HuffmanCoarse<ERRCTRL, HuffTrait<8>::type, MetadataTrait<4>::type>  //
        >;

    using FallbackCompressor = class DefaultPathCompressor<FallbackBinding>;
};

#endif