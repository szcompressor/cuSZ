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
#include "sp_path.cuh"
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
    using T = float;
    using E = ErrCtrlTrait<2>::type;
    using P = FastLowPrecisionTrait<true>::type;

    // TODO be part of the other two tasks without touching FS
    // if (ctx->task_is.experiment) {}

    if (ctx->task_is.construct or ctx->task_is.dryrun) {
        double time_loading{0.0};

        Capsule<T> in_data(ctx->data_len);
        in_data.alloc<cuszLOC::HOST_DEVICE, ALIGNDATA::SQUARE_MATRIX>()
            .from_fs_to<cuszLOC::HOST>(ctx->fnames.path2file, &time_loading)
            .host2device();

        if (ctx->verbose) LOGGING(LOG_DBG, "time loading datum:", time_loading, "sec");
        LOGGING(LOG_INFO, "load", ctx->fnames.path2file, ctx->data_len * sizeof(T), "bytes");

        Capsule<BYTE> out_dump;

        if (ctx->huff_bytewidth == 4) {
            DefaultPath::DefaultCompressor cuszc(ctx, &in_data);

            cuszc  //
                .compress()
                .consolidate<cuszLOC::HOST, cuszLOC::HOST>(&out_dump.get<cuszLOC::HOST>());
            cout << "output:\t" << ctx->fnames.compress_output << '\n';
            out_dump  //
                .to_fs_from<cuszLOC::HOST>(ctx->fnames.compress_output)
                .free<cuszLOC::HOST>();
        }
        else if (ctx->huff_bytewidth == 8) {
            DefaultPath::FallbackCompressor cuszc(ctx, &in_data);

            cuszc  //
                .compress()
                .consolidate<cuszLOC::HOST, cuszLOC::HOST>(&out_dump.get<cuszLOC::HOST>());
            cout << "output:\t" << ctx->fnames.compress_output << '\n';
            out_dump  //
                .to_fs_from<cuszLOC::HOST>(ctx->fnames.compress_output)
                .free<cuszLOC::HOST>();
        }
        else {
            throw std::runtime_error("huff nbyte illegal");
        }

        in_data.free<cuszLOC::HOST_DEVICE>();
    }

    if (ctx->task_is.reconstruct) {  // fp32 only for now

        auto fname_dump  = ctx->fnames.path2file + ".cusza";
        auto cusza_nbyte = ConfigHelper::get_filesize(fname_dump);

        Capsule<BYTE> in_dump(cusza_nbyte);
        in_dump  //
            .alloc<cuszLOC::HOST>()
            .from_fs_to<cuszLOC::HOST>(fname_dump);

        Capsule<T> out_xdata;

        // TODO try_writeback vs out_xdata.to_fs_from()
        if (ctx->huff_bytewidth == 4) {
            DefaultPath::DefaultCompressor cuszd(ctx, &in_dump);

            out_xdata  //
                .set_len(ctx->data_len)
                .alloc<cuszLOC::HOST_DEVICE, ALIGNDATA::SQUARE_MATRIX>();
            cuszd  //
                .decompress(&out_xdata)
                .backmatter(&out_xdata);
            out_xdata.free<cuszLOC::HOST_DEVICE>();
        }
        else if (ctx->huff_bytewidth == 8) {
            DefaultPath::FallbackCompressor cuszd(ctx, &in_dump);

            out_xdata  //
                .set_len(ctx->data_len)
                .alloc<cuszLOC::HOST_DEVICE, ALIGNDATA::SQUARE_MATRIX>();
            cuszd  //
                .decompress(&out_xdata)
                .backmatter(&out_xdata);
            out_xdata.free<cuszLOC::HOST_DEVICE>();
        }
    }
}

void special_path_spline3(cuszCTX* ctx)
{
    // TODO remove hardcode for float for now
    using T = float;
    using E = ErrCtrlTrait<4, true>::type;
    using P = FastLowPrecisionTrait<true>::type;

    using H_DUMMY = HuffTrait<4>::type;

    if (ctx->task_is.construct) {
        double time_loading{0.0};

        Capsule<T> in_data(ctx->data_len);
        in_data.alloc<cuszLOC::HOST_DEVICE>()
            .from_fs_to<cuszLOC::HOST>(ctx->fnames.path2file, &time_loading)
            .host2device();

        if (ctx->verbose) LOGGING(LOG_DBG, "time loading datum:", time_loading, "sec");
        LOGGING(LOG_INFO, "load", ctx->fnames.path2file, ctx->data_len * sizeof(T), "bytes");

        Capsule<BYTE> out_dump;

        SparsityAwarePath::DefaultCompressor cuszc(ctx, &in_data);
        cuszc  //
            .compress()
            .consolidate<cuszLOC::HOST, cuszLOC::HOST>(&out_dump.get<cuszLOC::HOST>());
        cout << "output:\t" << ctx->fnames.compress_output << '\n';
        out_dump  //
            .to_fs_from<cuszLOC::HOST>(ctx->fnames.compress_output)
            .free<cuszLOC::HOST>();

        in_data.free<cuszLOC::HOST_DEVICE>();
    }

    if (ctx->task_is.reconstruct) {  // fp32 only for now
                                     // TODO
        auto fname_dump  = ctx->fnames.path2file + ".cusza";
        auto cusza_nbyte = ConfigHelper::get_filesize(fname_dump);

        Capsule<BYTE> in_dump(cusza_nbyte);
        in_dump  //
            .alloc<cuszLOC::HOST>()
            .from_fs_to<cuszLOC::HOST>(fname_dump);

        Capsule<T> out_xdata;

        // TODO try_writeback vs out_xdata.to_fs_from()
        DefaultPath::DefaultCompressor cuszd(ctx, &in_dump);

        out_xdata  //
            .set_len(ctx->data_len)
            .alloc<cuszLOC::HOST_DEVICE>();
        cuszd  //
            .decompress(&out_xdata)
            .backmatter(&out_xdata);
        out_xdata.free<cuszLOC::HOST_DEVICE>();
    }
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
