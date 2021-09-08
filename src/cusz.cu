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
#include "kernel/preprocess.cuh"
#include "metadata.hh"
#include "nvgpusz.cuh"
#include "pack.hh"
#include "query.hh"
#include "type_aliasing.hh"
#include "types.hh"
#include "utils.hh"

double expectedErr;
double actualAbsErr;
double actualRelErr;
string z_mode;

namespace {

void load_demo_sizes(argpack* ap)
{
    std::unordered_map<std::string, std::vector<int>> dataset_entries = {
        {std::string("hacc"), {280953867, 1, 1, 1, 1}},    {std::string("hacc1b"), {1073726487, 1, 1, 1, 1}},
        {std::string("cesm"), {3600, 1800, 1, 1, 2}},      {std::string("hurricane"), {500, 500, 100, 1, 3}},
        {std::string("nyx-s"), {512, 512, 512, 1, 3}},     {std::string("nyx-m"), {1024, 1024, 1024, 1, 3}},
        {std::string("qmc"), {288, 69, 7935, 1, 3}},       {std::string("qmcpre"), {69, 69, 33120, 1, 3}},
        {std::string("exafel"), {388, 59200, 1, 1, 2}},    {std::string("rtm"), {235, 849, 849, 1, 3}},
        {std::string("parihaka"), {1168, 1126, 922, 1, 3}}};

    if (not ap->demo_dataset.empty()) {
        // TODO try-catch
        auto dim4_datum = dataset_entries.at(ap->demo_dataset);

        ap->dim4._0 = (int)dim4_datum[0];
        ap->dim4._1 = (int)dim4_datum[1];
        ap->dim4._2 = (int)dim4_datum[2];
        ap->dim4._3 = (int)dim4_datum[3];
        ap->ndim    = (int)dim4_datum[4];
    }

    if (ap->ndim == 1)
        ap->GPU_block_size = MetadataTrait<1>::Block;
    else if (ap->ndim == 2)
        ap->GPU_block_size = MetadataTrait<2>::Block;
    else if (ap->ndim == 3)
        ap->GPU_block_size = MetadataTrait<3>::Block;

    auto get_nblk = [&](auto d) { return (d + ap->GPU_block_size - 1) / ap->GPU_block_size; };

    ap->nblk4._0 = get_nblk(ap->dim4._0);
    ap->nblk4._1 = get_nblk(ap->dim4._1);
    ap->nblk4._2 = get_nblk(ap->dim4._2);
    ap->nblk4._3 = get_nblk(ap->dim4._3);

    ap->len = ap->dim4._0 * ap->dim4._1 * ap->dim4._2 * ap->dim4._3;

    ap->stride4 = {
        1,                          //
        ap->dim4._0,                //
        ap->dim4._0 * ap->dim4._1,  //
        ap->dim4._0 * ap->dim4._1 * ap->dim4._2};
}

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
    load_demo_sizes(ctx);

    if (ctx->verbose) {
        GetMachineProperties();
        GetDeviceProperty();
    }

    auto& subfiles = ctx->subfiles;

    // TODO hardcode for float for now
    using Data = float;

    auto len = ctx->len;
    auto m   = static_cast<size_t>(ceil(sqrt(len)));
    auto mxm = m * m;

    Capsule<Data> in_data(mxm);

    if (ctx->task_is.construct or ctx->task_is.dryrun) {
        logging(log_dbg, "add padding:", m, "units");

        cudaMalloc(&in_data.dptr, in_data.nbyte());
        cudaMallocHost(&in_data.hptr, in_data.nbyte());

        {
            auto a = hires::now();
            io::read_binary_to_array<Data>(subfiles.path2file, in_data.hptr, len);
            auto z = hires::now();
            logging(log_dbg, "time loading datum:", static_cast<duration_t>(z - a).count(), "sec");
            logging(log_info, "load", subfiles.path2file, len * sizeof(Data), "bytes");
        }

        in_data.h2d();

        if (ctx->mode == "r2r") {
            Analyzer analyzer;
            auto     result = analyzer.GetMaxMinRng                                     //
                          <Data, ExecutionPolicy::cuda_device, AnalyzerMethod::thrust>  //
                          (in_data.dptr, len);
            logging(log_dbg, "time scanning:", result.seconds, "sec");
            ctx->eb *= result.rng;
        }

        logging(
            log_dbg, std::to_string(ctx->quant_byte) + "-byte quant type,",
            std::to_string(ctx->huff_byte) + "-byte internal Huff type");
    }

    if (ctx->task_is.pre_binning) {
        cerr << log_err
             << "Binning is not working temporarily; we are improving end-to-end throughput by NOT touching "
                "filesystem. (ver. 0.1.4)"
             << endl;
        exit(1);
    }

    if (ctx->task_is.construct or ctx->task_is.dryrun) {  // fp32 only for now

        auto xyz = dim3(ctx->dim4._0, ctx->dim4._1, ctx->dim4._2);
        auto mp  = new cusz_header();

        if (ctx->quant_byte == 1) {
            if (ctx->huff_byte == 4)
                cusz_compress<true, 4, 1, 4>(ctx, &in_data, xyz, mp);
            else
                cusz_compress<true, 4, 1, 8>(ctx, &in_data, xyz, mp);
        }
        else if (ctx->quant_byte == 2) {
            if (ctx->huff_byte == 4)
                cusz_compress<true, 4, 2, 4>(ctx, &in_data, xyz, mp);
            else
                cusz_compress<true, 4, 2, 8>(ctx, &in_data, xyz, mp);
        }

        auto mp_byte = reinterpret_cast<char*>(mp);
        // yet another metadata package
        io::write_array_to_binary(subfiles.compress.out_yamp, mp_byte, sizeof(cusz_header));

        delete mp;

        // release memory
        cudaFree(in_data.dptr), cudaFreeHost(in_data.hptr);
    }

    if (in_data.dptr) {
        cudaFreeHost(in_data.dptr);  // really messy considering adp pointers are freed elsewhere
    }

    // invoke system() to untar archived files first before decompression
    if (not ctx->task_is.construct and ctx->task_is.reconstruct) {
        string cx_directory = subfiles.path2file.substr(0, subfiles.path2file.rfind('/') + 1);
        string cmd_string;
        if (cx_directory.length() == 0)
            cmd_string = "tar -xf " + subfiles.path2file + ".sz";
        else
            cmd_string = "tar -xf " + subfiles.path2file + ".sz" + " -C " + cx_directory;

        check_shell_calls(cmd_string);
    }

    if (ctx->task_is.reconstruct) {  // fp32 only for now

        // unpack metadata
        auto mp_byte = io::read_binary_to_new_array<char>(subfiles.decompress.in_yamp, sizeof(cusz_header));
        auto mp      = reinterpret_cast<cusz_header*>(mp_byte);

        if (ctx->quant_byte == 1) {
            if (ctx->huff_byte == 4)
                cusz_decompress<true, 4, 1, 4>(ctx, mp);
            else if (ctx->huff_byte == 8)
                cusz_decompress<true, 4, 1, 8>(ctx, mp);
        }
        else if (ctx->quant_byte == 2) {
            if (ctx->huff_byte == 4)
                cusz_decompress<true, 4, 2, 4>(ctx, mp);
            else if (ctx->huff_byte == 8)
                cusz_decompress<true, 4, 2, 8>(ctx, mp);
        }
    }

    // invoke system() function to merge and compress the resulting 5 files after cusz compression
    string basename = subfiles.path2file.substr(subfiles.path2file.rfind('/') + 1);
    if (not ctx->task_is.reconstruct and ctx->task_is.construct) {
        auto tar_a = hires::now();

        // remove *.sz if existing
        string cmd_string = "rm -rf " + ctx->opath + basename + ".sz";
        check_shell_calls(cmd_string);

        // using tar command to encapsulate files
        string files_to_merge;
        if (ctx->task_is.skip_huffman) {
            files_to_merge = basename + ".outlier " + basename + ".quant " + basename + ".yamp";
        }
        else {
            files_to_merge = basename + ".hbyte " + basename + ".outlier " + basename + ".canon " + basename +
                             ".hmeta " + basename + ".yamp";
        }
        if (ctx->task_is.lossless_gzip) {
            cmd_string = "cd " + ctx->opath + ";tar -czf " + basename + ".sz " + files_to_merge;
        }
        else {
            cmd_string = "cd " + ctx->opath + ";tar -cf " + basename + ".sz " + files_to_merge;
        }
        check_shell_calls(cmd_string);

        // remove 5 subfiles
        cmd_string = "cd " + ctx->opath + ";rm -rf " + files_to_merge;
        check_shell_calls(cmd_string);

        auto tar_z = hires::now();

        auto ad_hoc_fix = ctx->opath.substr(0, ctx->opath.size() - 1);
        logging(log_dbg, "time tar'ing:", static_cast<duration_t>(tar_z - tar_a).count(), "sec");
        logging(log_info, "output:", ad_hoc_fix + basename + ".sz");
    }

    // if it's decompression, remove released subfiles at last.
    if (not ctx->task_is.construct and ctx->task_is.reconstruct) {
        string files_to_delete;
        if (ctx->task_is.skip_huffman) {
            files_to_delete = basename + ".outlier " + basename + ".quant " + basename + ".yamp";
        }
        else {
            files_to_delete = basename + ".hbyte " + basename + ".outlier " + basename + ".canon " + basename +
                              ".hmeta " + basename + ".yamp";
        }
        string cmd_string =
            "cd " + subfiles.path2file.substr(0, subfiles.path2file.rfind('/')) + ";rm -rf " + files_to_delete;
        check_shell_calls(cmd_string);
    }

    if (ctx->task_is.construct and ctx->task_is.reconstruct) {
        // remove *.sz if existing
        string cmd_string = "rm -rf " + ctx->opath + basename + ".sz";
        check_shell_calls(cmd_string);

        // using tar command to encapsulate files
        string files_for_merging;
        if (ctx->task_is.skip_huffman) {
            files_for_merging = basename + ".outlier " + basename + ".quant " + basename + ".yamp";
        }
        else {
            files_for_merging = basename + ".hbyte " + basename + ".outlier " + basename + ".canon " + basename +
                                ".hmeta " + basename + ".yamp";
        }
        if (ctx->task_is.lossless_gzip) {
            cmd_string = "cd " + ctx->opath + ";tar -czf " + basename + ".sz " + files_for_merging;
        }
        else {
            cmd_string = "cd " + ctx->opath + ";tar -cf " + basename + ".sz " + files_for_merging;
        }
        check_shell_calls(cmd_string);

        // remove 5 subfiles
        cmd_string = "cd " + ctx->opath + ";rm -rf " + files_for_merging;
        check_shell_calls(cmd_string);

        logging(log_info, "write to: " + ctx->opath + basename + ".sz");
        logging(log_info, "write to: " + ctx->opath + basename + ".szx");

        /* gtest disabled in favor of code refactoring */
        // if (ap->task_is.gtest) {
        //     expectedErr  = ap->eb;
        //     z_mode       = ap->mode;
        //     auto stat    = ap->stat;
        //     actualAbsErr = stat.max_abserr;
        //     actualRelErr = stat.max_abserr_vs_rng;
        //     ::testing::InitGoogleTest(&argc, argv);
        //     return RUN_ALL_TESTS();
        // }
    }
}
