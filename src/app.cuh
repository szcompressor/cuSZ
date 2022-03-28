/**
 * @file app.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-02-20
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef APP_CUH
#define APP_CUH

#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../example/src/ex_common.cuh"
#include "analysis/analyzer.hh"
#include "common.hh"
#include "context.hh"
#include "default_path.cuh"
#include "query.hh"
#include "utils.hh"

using std::string;

namespace cusz {

template <typename Placeholder = float>
class app {
   private:
    using Header     = cuszHEADER;
    using Compressor = typename DefaultPath<float>::DefaultCompressor;
    using T          = typename Compressor::T;
    using Predictor  = typename Compressor::Predictor;
    using SpReducer  = typename Compressor::SpReducer;
    using Codec      = typename Compressor::Codec;

    const static auto HOST        = cusz::LOC::HOST;
    const static auto DEVICE      = cusz::LOC::DEVICE;
    const static auto HOST_DEVICE = cusz::LOC::HOST_DEVICE;

   public:
    using compressor_t = Compressor*;
    using context_t    = cuszCTX*;
    using header_t     = cuszHEADER*;

   public:
    template <class Predictor>
    static void cli_dryrun(cuszCTX* ctx, bool dualquant = true)
    {
        BaseCompressor<Predictor> analysis;

        uint3        xyz{ctx->x, ctx->y, ctx->z};
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        if (not dualquant) {
            analysis.init_dualquant_dryrun(xyz);
            analysis.dualquant_dryrun(ctx->fname.fname, ctx->eb, ctx->mode == "r2r", stream);
            analysis.destroy_dualquant_dryrun();
        }

        if (dualquant) {
            analysis.init_generic_dryrun(xyz);
            analysis.generic_dryrun(ctx->fname.fname, ctx->eb, 512, ctx->mode == "r2r", stream);
            analysis.destroy_generic_dryrun();
        }
        cudaStreamDestroy(stream);
    }

   private:
    static void init_detail(compressor_t* compressor, context_t ctx, header_t header)
    {
        if (ctx and header) throw std::runtime_error("init_compressor: two sources for configurations.");
        if ((not ctx) and (not header))
            throw std::runtime_error("init_compressor: neither source is for configurations.");

        if (ctx) {
            if (not *compressor) *compressor = new Compressor;
            // (get_xyz(ctx));
            AutoconfigHelper::autotune(ctx);
            (*compressor)->init(ctx);
        }
        if (header) {
            if (not *compressor) *compressor = new Compressor;
            // (get_xyz(header));
            (*compressor)->init(header);
        }
    }

   public:
    void init_compressor(context_t config)
    {
        if (not compressor) init_detail(&compressor, config, nullptr);
    }

    void init_compressor(header_t config)
    {
        if (not compressor) init_detail(&compressor, nullptr, config);
    }

    static void destroy_compressor(compressor_t compressor) { delete compressor; }

    void destroy_compressor() { delete compressor; }

   private:
    template <typename CONFIG>
    static dim3 get_xyz(CONFIG* c)
    {
        return dim3(c->x, c->y, c->z);
    }

    compressor_t compressor{nullptr};

   private:
    void cusz_write2disk_after_compress(BYTE* compressed, size_t compressed_len, string compressed_name)
    {
        Capsule<BYTE> file("cusza");
        file.set_len(compressed_len)
            .template set<DEVICE>(compressed)
            .template alloc<HOST>()
            .device2host()
            .to_file<HOST>(compressed_name)
            .template free<HOST_DEVICE>();
    }

   public:
    void cusz_write2disk_after_compress(string compressed_name)
    {
        cusz_write2disk_after_compress(compressed, compressed_len, compressed_name);
    }

   private:
    BYTE*  compressed{nullptr};
    size_t compressed_len{0};

   public:
    BYTE*  get_compressed() const { return compressed; }
    size_t get_len_compressed() const { return compressed_len; }
    void   get_compressed(BYTE*& out_compressed, size_t& out_compressed_len) const
    {
        out_compressed     = compressed;
        out_compressed_len = compressed_len;
    }

    /**
     * @brief high-level cusz_compress() API, exposing compressed results: `compressed` & `compressed_len`
     *
     * @param in_uncompressed input uncompressed data, with size known and embedded in params
     * @param params alias for a cusz context
     * @param compressed output 1, binary after compression
     * @param compressed_len output 2, size of the compressed binary
     * @param stream CUDA stream
     * @param report_time on-off, reporting kernel time
     */
    void cusz_compress(
        cuszCTX*     params,
        T*           in_uncompressed,
        BYTE*&       out_compressed,
        size_t&      out_compressed_len,
        cudaStream_t stream,
        bool         report_time = false)
    {
        (*compressor).compress(params, in_uncompressed, out_compressed, out_compressed_len, stream, report_time);
    }

   public:
    /**
     * @brief high-level cusz_compress() API, hiding: `compressed` & `compressed_len`
     *
     * @param in_uncompressed input uncompressed data, with size known and embedded in params
     * @param params alias for a cusz context
     * @param stream CUDA stream
     * @param report_time on-off, reporting kernel time
     */
    void cusz_compress(cuszCTX* params, T* in_uncompressed, cudaStream_t stream, bool report_time = false)
    {
        cusz_compress(params, in_uncompressed, compressed, compressed_len, stream, report_time);
    }

   public:
    void try_compare(cuszHEADER* header, Capsule<T>& xdata, Capsule<T>& cmp, string const& compare)
    {
        auto len             = (*header).get_len_uncompressed();
        auto compressd_bytes = (*header).file_size();

        auto compare_on_gpu = [&]() {
            cmp.template alloc<HOST_DEVICE>().template from_file<HOST>(compare).host2device();
            echo_metric_gpu(xdata.dptr, cmp.dptr, len, compressd_bytes);
            cmp.template free<HOST_DEVICE>();
        };

        auto compare_on_cpu = [&]() {
            cmp.template alloc<HOST>().template from_file<HOST>(compare);
            xdata.device2host();
            echo_metric_cpu(xdata.hptr, cmp.hptr, len, compressd_bytes);
            cmp.template free<HOST>();
        };

        if (compare != "") {
            float gb = 1.0 * sizeof(T) * len / 1e9;
            if (gb < 0.8)
                compare_on_gpu();
            else
                compare_on_cpu();
        }
    }

    void try_write(Capsule<T>& xdata, string basename, bool skip_write)
    {
        if (not skip_write) xdata.device2host().template to_file<HOST>(basename + ".cuszx");
    }

   public:
    /**
     * @brief high-level cusz_decompress() API
     *
     * @param in_compressed input compressed binary
     * @param params alias for cusz header
     * @param out_decompressed output decompressed data
     * @param stream CUDA stream
     * @param report_time on-off, reporting kernel time
     */
    void cusz_decompress(
        BYTE*        in_compressed,
        Header*      params,
        T*           out_decompressed,
        cudaStream_t stream,
        bool         report_time = false)
    {
        (*compressor).decompress(in_compressed, params, out_decompressed, stream, report_time);
    }

    template <typename T>
    static void input_shared(Capsule<T>& c, size_t len, T* from_hptr, T* from_dptr, std::string note = "input_<any>")
    {
        auto check_invalid = [&] {
            if ((not from_hptr) and (not from_dptr))
                throw std::runtime_error("input_uncompressed: must have one ptr from host or device");
            if ((not from_hptr) and (not from_dptr))
                throw std::runtime_error("input_uncompressed: must not have both ptrs from host and device");
        };

        check_invalid();

        if (from_dptr) c.set_len(len).template set<DEVICE>(from_dptr);
        if (from_hptr) c.set_len(len).template alloc<DEVICE>().template set<HOST>(from_hptr).host2device();
    }

    static void input_compressed(Capsule<BYTE>& compressed, size_t len, BYTE* from_hptr, BYTE* from_dptr)
    {
        input_shared<BYTE>(compressed, len, from_hptr, from_dptr);
    }

    template <typename T>
    static void input_uncompressed(Capsule<T>& uncompressed, size_t len, BYTE* from_hptr, BYTE* from_dptr)
    {
        input_shared<T>(uncompressed, len, from_hptr, from_dptr);
    }

    template <typename T>
    static void input_uncompressed(Capsule<T>& uncompressed, size_t len, std::string uncompressed_name)
    {
        double time_loading;
        uncompressed.set_len(len)
            .template alloc<HOST_DEVICE, cusz::ALIGNDATA::SQUARE_MATRIX>()
            .template from_file<HOST>(uncompressed_name, &time_loading)
            .host2device();
    }

    static void input_compressed(Capsule<BYTE>& compressed, std::string compressed_name)
    {
        auto compressed_len = ConfigHelper::get_filesize(compressed_name);
        compressed.set_len(compressed_len)
            .template alloc<HOST_DEVICE>()
            .template from_file<HOST>(compressed_name)
            .host2device();
    }

    /**
     * @brief a compressor dispatcher
     *
     * @param ctx context
     */
    void cusz_dispatch(cuszCTX* ctx)
    {
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        Capsule<T>    uncompressed("uncompressed");
        Capsule<BYTE> compressed("compressed");
        Capsule<T>    decompressed("decompressed"), cmp("cmp");

        auto basename = (*ctx).fname.fname;

        if ((*ctx).task_is.dryrun) cli_dryrun<Predictor>(ctx);
        if ((*ctx).task_is.construct) {  //
            auto len = (*ctx).x * (*ctx).y * (*ctx).z;

            input_uncompressed<T>(uncompressed, len, basename);
            if ((*ctx).mode == "r2r") (*ctx).eb *= uncompressed.prescan().get_rng();

            // core compression
            {
                init_compressor(ctx);
                cusz_compress(ctx, uncompressed.dptr, stream, (*ctx).report.time);
                cusz_write2disk_after_compress(basename + ".cusza");
            }
        }

        if ((*ctx).task_is.reconstruct) {
            // TODO improve header copy (GPU-CPU)
            auto header = new Header;
            input_compressed(compressed, basename + ".cusza");
            memcpy(header, compressed.hptr, sizeof(Header));

            auto len = (*header).get_len_uncompressed();
            decompressed.set_len(len).template alloc<HOST_DEVICE, cusz::ALIGNDATA::SQUARE_MATRIX>();
            cmp.set_len(len);

            // core decompression
            {
                init_compressor(header);
                cusz_decompress(compressed.dptr, header, decompressed.dptr, stream, (*ctx).report.time);

                try_compare(header, decompressed, cmp, (*ctx).fname.origin_cmp);
                try_write(decompressed, basename, (*ctx).to_skip.write2disk);
            }
        }

        if (stream) cudaStreamDestroy(stream);
    }
};

}  // namespace cusz

#endif
