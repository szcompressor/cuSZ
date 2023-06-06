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

template <class Combination>
class Compressor {
   public:
    using Spcodec = typename Combination::Spcodec;
    using Codec   = typename Combination::Codec;
    using BYTE    = uint8_t;
    /* removed, leaving vacancy in pipeline
     * using FallbackCodec = typename Combination::FallbackCodec;
     */

    using T  = typename Combination::DATA;
    using FP = typename Combination::FP;
    using E  = typename Combination::ERRCTRL;
    using H  = typename Codec::Encoded;
    using M  = typename Codec::MetadataT;
    // using H_FB = typename FallbackCodec::Encoded;

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

    Spcodec* spcodec;
    Codec*   codec;
    // variables
    uint32_t* d_freq;
    float     time_pred;
    float     time_hist;
    dim3      data_len3;
    size_t    _23june_datalen;

    // 23-june
    T*        _23june_d_anchor{nullptr};
    E*        _23june_d_errctrl{nullptr};
    T*        _23june_d_outlier{nullptr};
    uint32_t* _23june_d_outlier_idx{nullptr};

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
    // void encode_with_exception(E*, size_t, uint32_t*, int, int, int, bool, BYTE*&, size_t&, cudaStream_t, bool);
    void subfile_collect(T*, size_t, BYTE*, size_t, BYTE*, size_t, cudaStream_t, bool);
    void destroy();
    // getter
};

}  // namespace cusz

#endif
