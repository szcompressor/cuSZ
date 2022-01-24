/**
 * @file cusz.cu
 * @author Jiannan Tian
 * @brief Driver program of cuSZ.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2019-12-30
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

void cli_dryrun(cuszCTX* ctx, bool dualquant = true)
{
    BaseCompressor<DefaultPath::DefaultBinding::PREDICTOR> analysis;

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

void cli_compress(cuszCTX* ctx)
{
    double time_loading{0.0};

    Capsule<T> in_data(ctx->data_len, "comp::in_data");
    in_data.alloc<cusz::LOC::HOST_DEVICE, cusz::ALIGNDATA::SQUARE_MATRIX>()
        .from_file<cusz::LOC::HOST>(ctx->fname.fname, &time_loading)
        .host2device();

    if (ctx->verbose) LOGGING(LOG_DBG, "time loading datum:", time_loading, "sec");
    LOGGING(LOG_INFO, "load", ctx->fname.fname, ctx->data_len * sizeof(T), "bytes");

    Capsule<BYTE> out_archive("comp::out_archive");

    // TODO This does not cover the output size for *all* predictors.
    if (ctx->on_off.autotune_huffchunk)
        DefaultPath::DefaultBinding::CODEC::get_coarse_pardeg(ctx->data_len, ctx->huffman_chunksize, ctx->nchunk);
    else
        ctx->nchunk = ConfigHelper::get_npart(ctx->data_len, ctx->huffman_chunksize);

    uint3 xyz{ctx->x, ctx->y, ctx->z};

    if (ctx->huff_bytewidth == 4) {
        DefaultPath::DefaultCompressor cuszc(ctx, &in_data, xyz, ctx->dict_size);

        cuszc.compress(ctx->on_off.release_input)
            .consolidate<cusz::LOC::HOST, cusz::LOC::HOST>(&out_archive.get<cusz::LOC::HOST>());
        cout << "output:\t" << ctx->fname.compress_output << '\n';
        out_archive.to_file<cusz::LOC::HOST>(ctx->fname.compress_output).free<cusz::LOC::HOST>();
    }
    else if (ctx->huff_bytewidth == 8) {
        DefaultPath::FallbackCompressor cuszc(ctx, &in_data, xyz, ctx->dict_size);

        cuszc.compress(ctx->on_off.release_input)
            .consolidate<cusz::LOC::HOST, cusz::LOC::HOST>(&out_archive.get<cusz::LOC::HOST>());
        cout << "output:\t" << ctx->fname.compress_output << '\n';
        out_archive.to_file<cusz::LOC::HOST>(ctx->fname.compress_output).free<cusz::LOC::HOST>();
    }
    else {
        throw std::runtime_error("huff nbyte illegal");
    }
    if (ctx->on_off.release_input)
        in_data.free<cusz::LOC::HOST>();
    else
        in_data.free<cusz::LOC::HOST_DEVICE>();
}

void cli_decompress(cuszCTX* ctx)
{
    auto fin_archive = ctx->fname.fname + ".cusza";
    auto cusza_nbyte = ConfigHelper::get_filesize(fin_archive);

    Capsule<BYTE> in_archive(cusza_nbyte, ("decomp::in_archive"));
    in_archive.alloc<cusz::LOC::HOST>().from_file<cusz::LOC::HOST>(fin_archive);

    Capsule<T> out_xdata("decomp::out_xdata");

    // TODO try_writeback vs out_xdata.to_file()
    if (ctx->huff_bytewidth == 4) {
        DefaultPath::DefaultCompressor cuszd(ctx, &in_archive);
        out_xdata.set_len(ctx->data_len).alloc<cusz::LOC::HOST_DEVICE, cusz::ALIGNDATA::SQUARE_MATRIX>();
        cuszd.decompress(&out_xdata).backmatter(&out_xdata);
    }
    else if (ctx->huff_bytewidth == 8) {
        DefaultPath::FallbackCompressor cuszd(ctx, &in_archive);

        out_xdata.set_len(ctx->data_len).alloc<cusz::LOC::HOST_DEVICE, cusz::ALIGNDATA::SQUARE_MATRIX>();
        cuszd.decompress(&out_xdata).backmatter(&out_xdata);
    }
    out_xdata.free<cusz::LOC::HOST_DEVICE>();
}

void normal_path_lorenzo(cuszCTX* ctx)
{
    if (ctx->task_is.dryrun) cli_dryrun(ctx);
    if (ctx->task_is.construct) cli_compress(ctx);
    if (ctx->task_is.reconstruct) cli_decompress(ctx);
}

void special_path_spline3(cuszCTX* ctx)
{
    //
}

int main(int argc, char** argv)
{
    auto ctx = new cuszCTX(argc, argv);

    if (ctx->verbose) {
        GetMachineProperties();
        GetDeviceProperty();
    }

    if (ctx->str_predictor == "lorenzo") normal_path_lorenzo(ctx);
    if (ctx->str_predictor == "spline3") special_path_spline3(ctx);
}
