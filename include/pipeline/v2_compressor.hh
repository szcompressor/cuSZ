/**
 * @file v2_compressor.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-01-29
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>
#include <memory>

#include "common/type_traits.hh"
#include "compaction.hh"
#include "component.hh"
#include "context.hh"
#include "header.h"

// TODO move outward
#include "compaction_g.inl"

using Context = cusz::Context;

namespace psz {

template <class CONFIG>
class v2_Compressor {
   public:
    using BYTE = uint8_t;

    using T    = typename CONFIG::Predictor::Origin;
    using FP   = typename CONFIG::Predictor::Precision;
    using E    = typename CONFIG::Predictor::ErrCtrl;
    using H    = typename CONFIG::Codec::Encoded;
    using M    = typename CONFIG::Codec::MetadataT;
    using H_FB = typename CONFIG::FallbackCodec::Encoded;

    using TimeRecord   = std::vector<std::tuple<const char*, double>>;
    using timerecord_t = TimeRecord*;

   private:
    class impl;
    std::unique_ptr<impl> pimpl;

   public:
    ~v2_Compressor();
    v2_Compressor();
    v2_Compressor(const v2_Compressor&);
    v2_Compressor& operator=(const v2_Compressor&);
    v2_Compressor(v2_Compressor&&);
    v2_Compressor& operator=(v2_Compressor&&);

    // methods
    void init(Context*);
    void init(v2_header*);
    void destroy();
    void compress(Context*, T*, BYTE*&, size_t&, cudaStream_t = nullptr, bool = false);
    void decompress(v2_header*, BYTE*, T*, cudaStream_t = nullptr, bool = true);
    void clear_buffer();
    // getter
    void export_header(v2_header&);
    void export_header(v2_header*);
    void export_timerecord(TimeRecord*);
};

template <class CONFIG>
class v2_Compressor<CONFIG>::impl {
   public:
    using Codec = typename CONFIG::Codec;
    using BYTE  = uint8_t;
    using T     = typename CONFIG::Predictor::Origin;
    using FP    = typename CONFIG::Predictor::Precision;
    using EQ    = uint32_t;
    using H     = typename CONFIG::Codec::Encoded;
    using M     = uint32_t;
    using IDX   = uint32_t;
    using H_FB  = typename CONFIG::FallbackCodec::Encoded;

    using TimeRecord   = std::vector<std::tuple<const char*, double>>;
    using timerecord_t = TimeRecord*;

   private:
    // state
    // bool  use_fallback_codec{false};
    // bool  fallback_codec_allocated{false};

    BYTE* d_reserved_for_archive{nullptr};

    // profiling
    // TimeRecord timerecord;
    // header
    v2_header header;
    // components

    Codec* codec;

    // arrays
    T*                d_anchor;
    uint32_t*         d_errctrl;
    uint32_t*         d_freq;
    CompactionDRAM<T> outlier;

    int sp_factor{20};

    struct {
        float construct, hist, encode;
    } comp_time;

    struct {
        float scatter, decode, reconstruct;
    } decomp_time;

    dim3   data_len3;
    size_t data_len;

   public:
    ~impl();
    impl();

    // public methods
    void init(Context* config);
    void init(v2_header* config);

    void compress(Context*, T*, BYTE*&, size_t&, cudaStream_t = nullptr, bool = false);
    void decompress(v2_header*, BYTE*, T*, cudaStream_t = nullptr, bool = true);

    // getter
    void export_header(v2_header&);
    void export_header(v2_header*);
    // void export_timerecord(TimeRecord*);
    BYTE* var_archive() { return d_reserved_for_archive; };

   private:
    // helper
    template <class ContextOrHeader>
    void __init(ContextOrHeader*);

    // void collect_compress_timerecord();
    // void collect_decompress_timerecord();
    void destroy();
    // getter
};

}  // namespace psz
