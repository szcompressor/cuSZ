/**
 * @file compressor.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_COMPRESSOR_HH
#define CUSZ_COMPRESSOR_HH

#include <cuda_runtime.h>
#include <memory>

#include "common/type_traits.hh"
#include "compaction.hh"
#include "context.hh"
#include "header.h"
#include "hf/hf.hh"
// #include "prediction.inl"
// #include "spcodec.inl"

namespace cusz {

// extra helper
struct CompressorHelper {
    static int autotune_coarse_parvle(Context* ctx);
};

template <class BINDING>
class Compressor {
   public:
    using Predictor     = typename BINDING::Predictor;
    using Spcodec       = typename BINDING::Spcodec;
    using Codec         = typename BINDING::Codec;
    using FallbackCodec = typename BINDING::FallbackCodec;
    using BYTE          = uint8_t;

    using T    = typename Predictor::Origin;
    using FP   = typename Predictor::Precision;
    using E    = typename Predictor::ErrCtrl;
    using H    = typename Codec::Encoded;
    using M    = typename Codec::MetadataT;
    using H_FB = typename FallbackCodec::Encoded;

    using TimeRecord   = std::vector<std::tuple<const char*, double>>;
    using timerecord_t = TimeRecord*;

   private:
    // state
    bool  use_fallback_codec{false};
    bool  fallback_codec_allocated{false};
    BYTE* d_reserved_compressed{nullptr};
    // profiling
    TimeRecord timerecord;
    // header
    Header header;
    // components

    Predictor*     predictor;
    Spcodec*       spcodec;
    Codec*         codec;
    FallbackCodec* fb_codec;
    // variables
    uint32_t* d_freq;
    float     time_hist;
    dim3      data_len3;

   public:
    ~Compressor();
    Compressor();

    // public methods
    void init(Context* config, bool dbg_print = false);
    void init(Header* config, bool dbg_print = false);
    void compress(Context*, T*, BYTE*&, size_t&, cudaStream_t = nullptr, bool = false);
    void decompress(Header*, BYTE*, T*, cudaStream_t = nullptr, bool = true);
    void clear_buffer();

    // getter
    void     export_header(Header&);
    void     export_header(Header*);
    void     export_timerecord(TimeRecord*);
    uint32_t get_len_data();

   private:
    // helper
    template <class CONFIG>
    void init_detail(CONFIG*, bool);
    void init_codec(size_t, unsigned int, int, int, bool);
    void collect_compress_timerecord();
    void collect_decompress_timerecord();
    void encode_with_exception(E*, size_t, uint32_t*, int, int, int, bool, BYTE*&, size_t&, cudaStream_t, bool);
    void subfile_collect(T*, size_t, BYTE*, size_t, BYTE*, size_t, cudaStream_t, bool);
    void destroy();
    // getter
};

}  // namespace cusz

#endif
