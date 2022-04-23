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

#include "analysis/analyzer.hh"
#include "api.hh"
#include "cli/timerecord_viewer.hh"
#include "common.hh"
#include "compressor_impl.cuh"
#include "context.hh"
#include "query.hh"
#include "utils.hh"

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
        Header     header;
        auto       len      = (*ctx).get_len();
        auto       basename = (*ctx).fname.fname;

        auto load_uncompressed = [&](std::string fname) {
            input
                .set_len(len)  //
                .template alloc<HOST_DEVICE>(1.03)
                .template from_file<HOST>(fname)
                .host2device();
        };

        auto adjust_eb = [&]() {
            if ((*ctx).mode == "r2r") (*ctx).eb *= input.prescan().get_rng();
        };

        /******************************************************************************/

        load_uncompressed(basename);
        adjust_eb();

        TimeRecord timerecord;

        core_compress(compressor, ctx, input.dptr, len * 1.03, compressed, compressed_len, header, stream, &timerecord);

        if (ctx->report.time) TimeRecordViewer::view_compression(&timerecord, input.nbyte(), compressed_len);
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

        decompressed  //
            .set_len(len)
            .template alloc<HOST_DEVICE>(1.03);
        original.set_len(len);

        TimeRecord timerecord;

        core_decompress(
            compressor, header, compressed.dptr, header->get_filesize(), decompressed.dptr, len * 1.03, stream,
            &timerecord);

        if (ctx->report.time) TimeRecordViewer::view_decompression(&timerecord, decompressed.nbyte());
        QualityViewer::view(header, decompressed, original, (*ctx).fname.origin_cmp);
        try_write_decompressed_to_disk(decompressed, basename, (*ctx).skip.write2disk);
    }

   public:
    // TODO determine dtype & predictor in here
    void dispatch(context_t ctx)
    {
        auto predictor = (*ctx).predictor;
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
