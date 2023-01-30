/**
 * @file cli.cuh
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

#include "cli/analyzer.hh"
#include "cli/dryrun_part.cuh"
#include "cli/query.hh"
#include "cli/timerecord_viewer.hh"
#include "cusz.h"
#include "framework.hh"

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
    static void cli_dryrun(context_t ctx, bool dualquant = true)
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
    void write_compressed_to_disk(std::string compressed_name, BYTE* compressed, size_t compressed_len)
    {
        Capsule<BYTE> file("cusza");
        file.set_len(compressed_len)
            .template set<DEVICE>(compressed)
            .template alloc<HOST>()
            .device2host()
            .to_file<HOST>(compressed_name)
            .template free<HOST_DEVICE>();
    }

    void try_write_decompressed_to_disk(Capsule<T>& xdata, std::string basename, bool skip_write)
    {
        if (not skip_write) xdata.device2host().template to_file<HOST>(basename + ".cuszx");
    }

    // template <typename compressor_t>
    void cli_construct(context_t ctx, cusz_compressor* compressor, cudaStream_t stream)
    {
        Capsule<T> input("uncompressed");
        BYTE*      compressed;
        size_t     compressed_len;
        Header     header;
        auto       len      = ctx->get_len();
        auto       basename = ctx->fname.fname;

        auto load_uncompressed = [&](std::string fname) {
            input
                .set_len(len)  //
                .template alloc<HOST_DEVICE>()
                .template from_file<HOST>(fname)
                .host2device();
        };

        auto adjust_eb = [&]() {
            if (ctx->mode == "r2r") ctx->eb *= input.prescan().get_rng();
        };

        /******************************************************************************/

        load_uncompressed(basename);
        adjust_eb();

        TimeRecord timerecord;

        cusz_config* config     = new cusz_config{.eb = ctx->eb, .mode = Rel};
        cusz_len     uncomp_len = cusz_len{ctx->x, ctx->y, ctx->z, 1};

        cusz_compress(
            compressor, config, input.dptr, uncomp_len, &compressed, &compressed_len, &header, (void*)&timerecord,
            stream);

        if (ctx->report.time) TimeRecordViewer::view_compression(&timerecord, input.nbyte(), compressed_len);
        write_compressed_to_disk(basename + ".cusza", compressed, compressed_len);
    }

    // template <typename compressor_t>
    void cli_reconstruct(context_t ctx, cusz_compressor* compressor, cudaStream_t stream)
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
        auto len = ConfigHelper::get_uncompressed_len(header);

        decompressed  //
            .set_len(len)
            .template alloc<HOST_DEVICE>();
        original.set_len(len);

        TimeRecord timerecord;

        cusz_len decomp_len = cusz_len{header->x, header->y, header->z, 1};

        cusz_decompress(
            compressor, header, compressed.dptr, ConfigHelper::get_filesize(header), decompressed.dptr, decomp_len,
            (void*)&timerecord, stream);

        if (ctx->report.time) TimeRecordViewer::view_decompression(&timerecord, decompressed.nbyte());
        QualityViewer::view(header, decompressed, original, (*ctx).fname.origin_cmp);
        try_write_decompressed_to_disk(decompressed, basename, (*ctx).skip.write2disk);

        decompressed.template free<HOST_DEVICE>();
    }

   public:
    // TODO determine dtype & predictor in here
    void dispatch(context_t ctx)
    {
        // TODO disable predictor selection; to specify in another way
        // auto predictor = (*ctx).predictor;

        cusz_framework*  framework  = cusz_default_framework();
        cusz_compressor* compressor = cusz_create(framework, FP32);

        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));

        // TODO hardcoded predictor type
        if ((*ctx).cli_task.dryrun) cli_dryrun<typename Framework<float>::Predictor>(ctx);

        if ((*ctx).cli_task.construct) cli_construct(ctx, compressor, stream);

        if ((*ctx).cli_task.reconstruct) cli_reconstruct(ctx, compressor, stream);

        if (stream) cudaStreamDestroy(stream);
    }
};

}  // namespace cusz

#endif
