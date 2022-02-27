/**
 * @file cusz.cu
 * @author Jiannan Tian
 * @brief Driver program of cuSZ.
 * @version 0.1
 * @date 2020-09-20
 * (created) 2019-12-30 (rev) 2022-02-20
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

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

using std::string;

#include "../example/src/ex_common.cuh"
#include "analysis/analyzer.hh"
#include "common.hh"
#include "context.hh"
#include "default_path.cuh"
#include "header.hh"
#include "query.hh"
#include "utils.hh"

namespace {

template <typename T>
void check_shell_calls(string cmd_string)
{
    char* cmd = new char[cmd_string.length() + 1];
    strcpy(cmd, cmd_string.c_str());
    int status = system(cmd);
    delete[] cmd;
    cmd = nullptr;
    if (status < 0) { LOGGING(LOG_ERR, "Shell command call failed, exit code: ", errno, "->", strerror(errno)); }
}

}  // namespace

template <typename T, int DownscaleFactor, int tBLK>
T* pre_binning(T* d, size_t* dim_array)
{
    throw std::runtime_error("[pre_binning] disabled temporarily, will be part of preprocessing.");
    return nullptr;
}

using T = float;
using E = ErrCtrlTrait<2>::type;
using P = FastLowPrecisionTrait<true>::type;

template <class Predictor>
void cli_dryrun(cuszCTX* ctx, bool dualquant = true)
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

template <class Compressor>
void defaultpath_compress(
    cuszCTX*                         ctx,
    cudaStream_t                     stream,
    Capsule<typename Compressor::T>* in_uncompressed,
    std::string const                basename)
{
    using Predictor = typename Compressor::Predictor;
    using SpReducer = typename Compressor::SpReducer;
    using Codec     = typename Compressor::Codec;
    using Header    = typename Compressor::HEADER;
    using T         = typename Compressor::T;

    auto autotune = [&]() -> int {
        // TODO should be move to somewhere else, e.g., cusz::par_optmizer
        if (ctx->on_off.autotune_huffchunk)
            Compressor::Codec::get_coarse_pardeg(ctx->data_len, ctx->huffman_chunksize, ctx->nchunk);
        else
            ctx->nchunk = ConfigHelper::get_npart(ctx->data_len, ctx->huffman_chunksize);

        return ctx->nchunk;
    };

    BYTE*       compressed;
    size_t      compressed_len;
    std::string compressed_name = basename + ".cusza";

    Capsule<BYTE> file("cusza");

    int  radius = (*ctx).radius;
    auto xyz    = dim3((*ctx).x, (*ctx).y, (*ctx).z);
    auto pardeg = autotune();
    auto data   = *in_uncompressed;

    auto adjustd_eb = (*ctx).eb;
    auto r2r        = (*ctx).mode == "r2r";
    if (r2r) adjustd_eb *= data.prescan().get_rng();
    auto force_use_fallback_codec = (*ctx).huff_bytewidth == 8;

    Compressor compressor(xyz);
    compressor.allocate_workspace(radius, pardeg);  // alpha: overallocate for decompresison

    compressor.compress(
        data.dptr, adjustd_eb, radius, pardeg, compressed, compressed_len, force_use_fallback_codec, stream,
        (*ctx).report.time);

    file.set_len(compressed_len)
        .template set<cusz::LOC::DEVICE>(compressed)
        .template alloc<cusz::LOC::HOST>()
        .device2host()
        .to_file<cusz::LOC::HOST>(compressed_name)
        .template free<cusz::LOC::HOST_DEVICE>();
}

template <class Compressor>
void defaultpath_decompress(
    typename Compressor::HEADER* header,
    cudaStream_t                 stream,
    Capsule<BYTE>*               in_compressed,
    std::string const&           basename,
    std::string const&           compare,
    bool                         rpt_print,
    bool                         skip_write)
{
    using Header = typename Compressor::HEADER;
    using T      = typename Compressor::T;

    Capsule<T> xdata("xdata"), cmp("cmp");
    auto       x = (*header).x, y = (*header).y, z = (*header).z;
    auto       xyz    = dim3(x, y, z);
    auto       len    = x * y * z;
    auto       eb     = (*header).eb;
    auto       radius = (*header).radius;
    auto       pardeg = (*header).vle_pardeg;  // TODO don't really need pardeg in here

    auto try_compare = [&]() {
        if (compare != "") {
            float gb              = 1.0 * sizeof(T) * len / 1e9;
            auto  compressd_bytes = (*header).file_size();

            if (gb < 0.8) {
                cmp.template alloc<cusz::LOC::HOST_DEVICE>().template from_file<cusz::LOC::HOST>(compare).host2device();
                echo_metric_gpu(xdata.dptr, cmp.dptr, len, compressd_bytes);
                cmp.template free<cusz::LOC::HOST_DEVICE>();
            }
            else {
                cmp.template alloc<cusz::LOC::HOST>().template from_file<cusz::LOC::HOST>(compare);
                xdata.device2host();
                echo_metric_cpu(xdata.hptr, cmp.hptr, len, compressd_bytes);
                cmp.template free<cusz::LOC::HOST>();
            }
        }
    };

    auto try_write = [&]() {
        if (not skip_write) xdata.device2host().template to_file<cusz::LOC::HOST>(basename + ".cuszx");
    };

    xdata.set_len(len).template alloc<cusz::LOC::HOST_DEVICE, cusz::ALIGNDATA::SQUARE_MATRIX>();
    cmp.set_len(len).set_name("origin-cmp");

    Compressor compressor(xyz);
    compressor.decompress((*in_compressed).dptr, eb, radius, xdata.dptr, stream, rpt_print);

    try_compare();
    try_write();
}

void defaultpath(cuszCTX* ctx)
{
    using T         = DefaultPath::DefaultCompressor::T;
    using Header    = DefaultPath::DefaultCompressor::HEADER;
    using Predictor = DefaultPath::DefaultCompressor::Predictor;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    Capsule<T> data("data");

    auto basename = (*ctx).fname.fname;

    if ((*ctx).task_is.dryrun) cli_dryrun<Predictor>(ctx);

    if ((*ctx).task_is.construct) {  //
        auto   len = (*ctx).x * (*ctx).y * (*ctx).z;
        double time_loading{0.0};

        Capsule<T> data;
        data.set_len(len)
            .template alloc<cusz::LOC::HOST_DEVICE, cusz::ALIGNDATA::SQUARE_MATRIX>()
            .template from_file<cusz::LOC::HOST>(basename, &time_loading)
            .host2device();

        defaultpath_compress<DefaultPath::DefaultCompressor>(ctx, stream, &data, basename);
    }

    if ((*ctx).task_is.reconstruct) {
        auto compressed_name = basename + ".cusza";
        auto compressed_len  = ConfigHelper::get_filesize(compressed_name);
        auto cmp_name        = (*ctx).fname.origin_cmp;

        Capsule<BYTE> file;
        file.set_len(compressed_len)
            .template alloc<cusz::LOC::HOST_DEVICE>()
            .template from_file<cusz::LOC::HOST>(compressed_name)
            .host2device();

        Header header;
        memcpy(&header, file.hptr, sizeof(Header));

        auto skip_write = (*ctx).to_skip.write2disk;

        defaultpath_decompress<DefaultPath::DefaultCompressor>(
            &header, stream, &file, basename, cmp_name, (*ctx).report.time, skip_write);
    }

    if (stream) cudaStreamDestroy(stream);
}

void sparsitypath_spline3(cuszCTX* ctx)
{
    // TODO
}

int main(int argc, char** argv)
{
    auto ctx = new cuszCTX(argc, argv);

    if (ctx->verbose) {
        GetMachineProperties();
        GetDeviceProperty();
    }

    if (ctx->str_predictor == "lorenzo") defaultpath(ctx);
    if (ctx->str_predictor == "spline3") sparsitypath_spline3(ctx);
}
