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
#include "header.hh"
#include "kernel/preprocess.cuh"
#include "nvgpusz.cuh"
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

#define NONPTR_TYPE(VAR) std::remove_pointer<decltype(VAR)>::type

void normal_path_lorenzo(cuszCTX* ctx)
{
    // TODO remove hardcode for float for now
    using T = float;
    using E = QuantTrait<2>::type;
    using P = FastLowPrecisionTrait<true>::type;

    // TODO align to 128 unconditionally
    auto len = ctx->data_len;
    auto m   = Reinterpret1DTo2D::get_square_size(len);
    auto mxm = m * m;

    Capsule<T> in_data(mxm);

    if (ctx->task_is.construct or ctx->task_is.experiment or ctx->task_is.dryrun) {
        cudaMalloc(&in_data.dptr, in_data.nbyte());
        cudaMallocHost(&in_data.hptr, in_data.nbyte());

        {
            auto a = hires::now();
            io::read_binary_to_array<T>(ctx->fnames.path2file, in_data.hptr, len);
            auto z = hires::now();

            if (ctx->verbose) LOGGING(LOG_DBG, "time loading datum:", static_cast<duration_t>(z - a).count(), "sec");

            LOGGING(LOG_INFO, "load", ctx->fnames.path2file, len * sizeof(T), "bytes");
        }

        in_data.h2d();

        if (ctx->mode == "r2r") {
            // TODO prescan can be issued independently from "r2r"
            auto result = Analyzer::get_maxmin_rng                         //
                <T, ExecutionPolicy::cuda_device, AnalyzerMethod::thrust>  //
                (in_data.dptr, len);
            if (ctx->verbose) LOGGING(LOG_DBG, "time scanning:", result.seconds, "sec");
            if (ctx->mode == "r2r") ctx->eb *= result.rng;
        }

        if (ctx->verbose)
            LOGGING(
                LOG_DBG, std::to_string(ctx->quant_nbyte) + "-byte quant type,",
                std::to_string(ctx->huff_nbyte) + "-byte (internal) Huff type");
    }

    if (ctx->preprocess.binning) {
        LOGGING(
            LOG_ERR,
            "Binning is not working temporarily; we are improving end-to-end throughput by NOT touching filesystem. "
            "(ver. 0.2.9)");
        exit(1);
    }

    if (ctx->task_is.construct or ctx->task_is.dryrun) {  // fp32 only for now

        if (ctx->huff_nbyte == 4) {
            Compressor<T, E, HuffTrait<4>::type, P> cuszc(ctx, cusz::WHEN::COMPRESS);
            cuszc.compress(&in_data);
        }
        else {
            Compressor<T, E, HuffTrait<8>::type, P> cuszc(ctx, cusz::WHEN::COMPRESS);
            cuszc.compress(&in_data);
        }

        // release memory
        cudaFree(in_data.dptr), cudaFreeHost(in_data.hptr);
    }

    if (in_data.dptr) {
        cudaFreeHost(in_data.dptr);  // TODO messy
    }

    if (ctx->task_is.reconstruct) {  // fp32 only for now

        // TODO data ready outside Decompressor?

        if (ctx->huff_nbyte == 4) {
            Compressor<T, E, HuffTrait<4>::type, P> cuszd(ctx, cusz::WHEN::DECOMPRESS);
            cuszd.decompress();
        }
        else if (ctx->huff_nbyte == 8) {
            Compressor<T, E, HuffTrait<8>::type, P> cuszd(ctx, cusz::WHEN::DECOMPRESS);
            cuszd.decompress();
        }
        // }
    }
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

    if (ctx->predictor == "lorenzo") normal_path_lorenzo(ctx);
    if (ctx->predictor == "spline3") special_path_spline3(ctx);
}
