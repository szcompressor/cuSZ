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
#include "argparse.hh"
#include "capsule.hh"
#include "header.hh"
#include "kernel/preprocess.cuh"
#include "metadata.hh"
#include "nvgpusz.cuh"
#include "query.hh"
#include "type_aliasing.hh"
#include "types.hh"
#include "utils.hh"

// double expectedErr;
// double actualAbsErr;
// double actualRelErr;
// string z_mode;

namespace {

template <typename Data>
void check_shell_calls(string cmd_string)
{
    char* cmd = new char[cmd_string.length() + 1];
    strcpy(cmd, cmd_string.c_str());
    int status = system(cmd);
    delete[] cmd;
    cmd = nullptr;
    if (status < 0) { logging(log_err, "Shell command call failed, exit code: ", errno, "->", strerror(errno)); }
}

}  // namespace

/* gtest disabled in favor of code refactoring */
// TEST(cuSZTest, TestMaxError)
// {
//     double actualErr = (z_mode == "r2r") ? actualRelErr : actualAbsErr;
//     ASSERT_LE(actualErr, expectedErr);
// }

template <typename Data, int DownscaleFactor, int tBLK>
Data* pre_binning(Data* d, size_t* dim_array)
{
    throw std::runtime_error("[pre_binning] disabled temporarily, will be part of preprocessing.");
    return nullptr;
}

#define NONPTR_TYPE(VAR) std::remove_pointer<decltype(VAR)>::type

int main(int argc, char** argv)
{
    auto ctx = new ArgPack();
    ctx->parse_args(argc, argv);

    if (ctx->verbose) {
        GetMachineProperties();
        GetDeviceProperty();
    }

    // TODO remove hardcode for float for now
    using Data = float;

    auto len = ctx->data_len;
    auto m   = static_cast<size_t>(ceil(sqrt(len)));
    auto mxm = m * m;

    Capsule<Data> in_data(mxm);

    if (ctx->task_is.construct or ctx->task_is.dryrun) {
        // logging(log_dbg, "add padding:", m, "units");

        cudaMalloc(&in_data.dptr, in_data.nbyte());
        cudaMallocHost(&in_data.hptr, in_data.nbyte());

        {
            auto a = hires::now();
            io::read_binary_to_array<Data>(ctx->fnames.path2file, in_data.hptr, len);
            auto z = hires::now();

            if (ctx->verbose) logging(log_dbg, "time loading datum:", static_cast<duration_t>(z - a).count(), "sec");

            logging(log_info, "load", ctx->fnames.path2file, len * sizeof(Data), "bytes");
        }

        in_data.h2d();

        if (ctx->mode == "r2r") {
            Analyzer analyzer;
            auto     result = analyzer.GetMaxMinRng                                     //
                          <Data, ExecutionPolicy::cuda_device, AnalyzerMethod::thrust>  //
                          (in_data.dptr, len);
            if (ctx->verbose) logging(log_dbg, "time scanning:", result.seconds, "sec");
            ctx->eb *= result.rng;
        }

        if (ctx->verbose)
            logging(
                log_dbg, std::to_string(ctx->quant_nbyte) + "-byte quant type,",
                std::to_string(ctx->huff_nbyte) + "-byte internal Huff type");
    }

    if (ctx->task_is.pre_binning) {
        cerr << log_err
             << "Binning is not working temporarily; we are improving end-to-end throughput by NOT touching "
                "filesystem. (ver. 0.1.4)"
             << endl;
        exit(1);
    }

    if (ctx->task_is.construct or ctx->task_is.dryrun) {  // fp32 only for now

        if (ctx->quant_nbyte == 1) {
            throw runtime_error("Quant=1-byte temporarily disabled.");
            if (ctx->huff_nbyte == 4) {
                // cusz_compress<true, 4, 1, 4>(ctx, &in_data);
            }
            else {
                // cusz_compress<true, 4, 1, 8>(ctx, &in_data);
            }
        }
        else if (ctx->quant_nbyte == 2) {
            if (ctx->huff_nbyte == 4) {  //
                cusz_compress<true, 4, 2, 4>(ctx, &in_data);
            }
            else {
                cusz_compress<true, 4, 2, 8>(ctx, &in_data);
            }
        }

        // release memory
        cudaFree(in_data.dptr), cudaFreeHost(in_data.hptr);
    }

    if (in_data.dptr) {
        cudaFreeHost(in_data.dptr);  // TODO messy
    }

    if (ctx->task_is.reconstruct) {  // fp32 only for now

        // TODO data ready outside Decompressor?

        if (ctx->quant_nbyte == 1) {
            throw runtime_error("Quant=1-byte temporarily disabled.");
            if (ctx->huff_nbyte == 4) {
                // cusz_decompress<true, 4, 1, 4>(ctx);
            }
            else if (ctx->huff_nbyte == 8) {
                // cusz_decompress<true, 4, 1, 8>(ctx);
            }
        }
        else if (ctx->quant_nbyte == 2) {
            if (ctx->huff_nbyte == 4)
                cusz_decompress<true, 4, 2, 4>(ctx);
            else if (ctx->huff_nbyte == 8)
                cusz_decompress<true, 4, 2, 8>(ctx);
        }
    }
}
