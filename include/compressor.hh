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
    /* removed, leaving vacancy in pipeline
     * using FallbackCodec = typename Combination::FallbackCodec;
     * using Spcodec = typename Combination::Spcodec;
     */

    using Codec = typename Combination::Codec;
    using BYTE  = uint8_t;

    using T  = typename Combination::DATA;
    using FP = typename Combination::FP;
    using E  = typename Combination::ERRCTRL;
    using H  = typename Codec::Encoded;
    using M  = typename Codec::MetadataT;
    // using H_FB = typename FallbackCodec::Encoded;

    using TimeRecord   = std::vector<std::tuple<const char*, double>>;
    using timerecord_t = TimeRecord*;

   private:
    // profiling
    TimeRecord timerecord;

    // header
    Header header;

    // external codec that has complex internals
    Codec* codec;

    float time_pred, time_hist, time_sp;

    // sizes
    dim3   data_len3;
    size_t datalen_linearized;
    int    splen;

    // configs
    float _23june_density{0.2};
    bool  use_fallback_codec{false};        // obsolete
    bool  fallback_codec_allocated{false};  // obsolete

    // buffers
    BYTE*     d_reserved_compressed{nullptr};
    E*        d_errctrl{nullptr};  // pred out1
    T*        d_outlier{nullptr};  // pred out2
    uint32_t* d_freq;              // hist out
    T*        d_spval{nullptr};    // gather out1
    uint32_t* d_spidx{nullptr};    // gather out2

   public:
    Compressor() = default;
    ~Compressor();

    // public methods
    void init(Context* config, bool dbg_print = false);
    void init(Header* config, bool dbg_print = false);
    void compress(Context*, T*, BYTE*&, size_t&, cudaStream_t = nullptr, bool = false);
    void decompress(Header*, BYTE*, T*, cudaStream_t = nullptr, bool = true);
    void clear_buffer();

    // getter
    void export_header(Header&);
    void export_header(Header*);
    void export_timerecord(TimeRecord*);

   private:
    // helper
    template <class CONFIG>
    void init_detail(CONFIG*, bool);
    void collect_compress_timerecord();
    void collect_decompress_timerecord();
    void merge_subfiles(BYTE*, size_t, T*, M*, size_t, cudaStream_t);
    void destroy();
    // getter
};

}  // namespace cusz

#endif
