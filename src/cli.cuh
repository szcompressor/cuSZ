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

#ifndef CLI_CUH
#define CLI_CUH

#include <string>
#include <type_traits>

#include "../example/src/ex_common.cuh"
#include "analysis/analyzer.hh"
#include "api.hh"
#include "common.hh"
#include "context.hh"
#include "compressor.cuh"
#include "query.hh"
#include "utils.hh"

using std::string;

namespace cusz {

template <typename Data = float>
class CLI {
   private:
    using Header = cuszHEADER;
    using T      = Data;

    const static auto HOST        = cusz::LOC::HOST;
    const static auto DEVICE      = cusz::LOC::DEVICE;
    const static auto HOST_DEVICE = cusz::LOC::HOST_DEVICE;

    using context_t = cuszCTX*;
    using header_t  = cuszHEADER*;

   public:
    CLI() = default;

    template <class Predictor>
    static void dryrun(context_t ctx, bool dualquant = true)
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
        else {
            analysis.init_generic_dryrun(xyz);
            analysis.generic_dryrun(ctx->fname.fname, ctx->eb, 512, ctx->mode == "r2r", stream);
            analysis.destroy_generic_dryrun();
        }
        cudaStreamDestroy(stream);
    }

   private:
    void write_compressed_to_disk(string compressed_name, BYTE* compressed, size_t compressed_len)
    {
        Capsule<BYTE> file("cusza");
        file.set_len(compressed_len)
            .template set<DEVICE>(compressed)
            .template alloc<HOST>()
            .device2host()
            .to_file<HOST>(compressed_name)
            .template free<HOST_DEVICE>();
    }

    void try_evaluate_quality(header_t header, Capsule<T>& xdata, Capsule<T>& cmp, string const& compare)
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
            auto gb = 1.0 * sizeof(T) * len / 1e9;
            if (gb < 0.8)
                compare_on_gpu();
            else
                compare_on_cpu();
        }
    }

    void try_write_decompressed_to_disk(Capsule<T>& xdata, string basename, bool skip_write)
    {
        if (not skip_write) xdata.device2host().template to_file<HOST>(basename + ".cuszx");
    }

    template <typename compressor_t>
    void construct(context_t ctx, compressor_t compressor, cudaStream_t stream)
    {
        Capsule<T> input("uncompressed");
        BYTE*      compressed;
        size_t     compressed_len;
        header_t   header;
        auto       len      = (*ctx).get_len();
        auto       basename = (*ctx).fname.fname;

        auto load_uncompressed = [&](std::string fname) {
            input.set_len(len)
                .template alloc<HOST_DEVICE, cusz::ALIGNDATA::SQUARE_MATRIX>()
                .template from_file<HOST>(fname)
                .host2device();
        };

        auto adjust_eb = [&]() {
            if ((*ctx).mode == "r2r") (*ctx).eb *= input.prescan().get_rng();
        };

        /******************************************************************************/

        load_uncompressed(basename);
        adjust_eb();

        core_compress(ctx, compressor, input.dptr, compressed, compressed_len, header, stream, (*ctx).report.time);

        write_compressed_to_disk(basename + ".cusza", compressed, compressed_len);
    }

    template <typename compressor_t>
    void reconstruct(context_t ctx, compressor_t compressor, cudaStream_t stream)
    {
        Capsule<BYTE> compressed("compressed");
        Capsule<T>    decompressed("decompressed"), original("cmp");
        auto          header   = new Header;
        auto          basename = (*ctx).fname.fname;

        auto load_compressed = [&](std::string compressed_name) {
            auto compressed_len = ConfigHelper::get_filesize(compressed_name);
            compressed.set_len(compressed_len)
                .template alloc<HOST_DEVICE>()
                .template from_file<HOST>(compressed_name)
                .host2device();
        };

        /******************************************************************************/

        load_compressed(basename + ".cusza");
        memcpy(header, compressed.hptr, sizeof(Header));
        auto len = (*header).get_len_uncompressed();

        decompressed.set_len(len).template alloc<HOST_DEVICE, cusz::ALIGNDATA::SQUARE_MATRIX>();
        original.set_len(len);

        core_decompress(header, compressor, compressed.dptr, decompressed.dptr, stream, (*ctx).report.time);

        try_evaluate_quality(header, decompressed, original, (*ctx).fname.origin_cmp);
        try_write_decompressed_to_disk(decompressed, basename, (*ctx).to_skip.write2disk);
    }

   public:
    // TODO determine dtype & predictor in here
    void dispatch(context_t ctx)
    {
        auto predictor = (*ctx).str_predictor;
        if (predictor == "lorenzo") {
            using Compressor = typename Framework<Data>::LorenzoFeaturedCompressor;
            dispatch_task<Compressor>(ctx);
        }
        else if (predictor == "spline3") {
            using Compressor = typename Framework<Data>::Spline3FeaturedCompressor;
            throw std::runtime_error("Spline3 based compressor is not ready.");
            // dispatch_task<Compressor>(ctx);
        }
        else {
            using Compressor = typename Framework<Data>::DefaultCompressor;
            dispatch_task<Compressor>(ctx);
        }
    }

   private:
    template <class Compressor>
    void dispatch_task(context_t ctx)
    {
        using Predictor = typename Compressor::Predictor;
        auto compressor = new Compressor;

        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));

        if ((*ctx).task_is.dryrun) dryrun<Predictor>(ctx);

        if ((*ctx).task_is.construct) construct(ctx, compressor, stream);

        if ((*ctx).task_is.reconstruct) reconstruct(ctx, compressor, stream);

        if (stream) cudaStreamDestroy(stream);
    }
};

}  // namespace cusz

#endif
