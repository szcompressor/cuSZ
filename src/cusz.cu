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
#include <unordered_map>
#include <vector>

using std::string;

#if __cplusplus >= 201103L

#include "analysis/analyzer.hh"
#include "argparse.hh"
#include "cusz_interface.cuh"
#include "datapack.hh"
#include "filter.cuh"
#include "gtest/gtest.h"
#include "metadata.hh"
#include "pack.hh"
#include "query.hh"
#include "type_aliasing.hh"
#include "types.hh"
#include "utils/cuda_err.cuh"
#include "utils/cuda_mem.cuh"
#include "utils/format.hh"
#include "utils/io.hh"
#include "utils/timer.hh"

double expectedErr;
double actualAbsErr;
double actualRelErr;
string z_mode;

void InitializeDims(argpack* ap)
{
    std::unordered_map<std::string, std::vector<int>> dataset_entries = {
        {std::string("hacc"), {280953867, 1, 1, 1, 1}},    {std::string("hacc1b"), {1073726487, 1, 1, 1, 1}},
        {std::string("cesm"), {3600, 1800, 1, 1, 2}},      {std::string("hurricane"), {500, 500, 100, 1, 3}},
        {std::string("nyx-s"), {512, 512, 512, 1, 3}},     {std::string("nyx-m"), {1024, 1024, 1024, 1, 3}},
        {std::string("qmc"), {288, 69, 7935, 1, 3}},       {std::string("qmcpre"), {69, 69, 33120, 1, 3}},
        {std::string("exafel"), {388, 59200, 1, 1, 2}},    {std::string("aramco"), {235, 849, 849, 1, 3}},
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

void CheckShellCall(string cmd_string)
{
    char* cmd = new char[cmd_string.length() + 1];
    strcpy(cmd, cmd_string.c_str());
    int status = system(cmd);
    delete[] cmd;
    cmd = nullptr;
    if (status < 0) { LogAll(log_err, "Shell command call failed, exit code: ", errno, "->", strerror(errno)); }
}

TEST(cuSZTest, TestMaxError)
{
    double actualErr = (z_mode == "r2r") ? actualRelErr : actualAbsErr;
    ASSERT_LE(actualErr, expectedErr);
}

template <typename Data, int DownscaleFactor, int tBLK>
Data* pre_binning(Data* d, size_t* dim_array)
{
    return nullptr;
}

int main(int argc, char** argv)
{
    cout << "\n>>>>  cusz build: 2020-04-29.0\n\n";

    auto ap = new ArgPack();
    ap->ParseCuszArgs(argc, argv);

    int    nnz_outlier = 0;
    size_t total_bits, total_uInt, huff_meta_size;
    bool   nvcomp_in_use = false;

    if (ap->verbose) {
        GetMachineProperties();
        GetDeviceProperty();
    }

    auto& wf       = ap->szwf;
    auto& subfiles = ap->subfiles;

    // TODO hardcode for float for now
    using DataInUse = float;
    DataPack<DataInUse> datapack{};
    DataInUse*          data = nullptr;

    if (wf.lossy_construct or wf.lossy_dryrun) {
        InitializeDims(ap);

        LogAll(
            log_info, "load", subfiles.path2file, ap->len * (ap->dtype == "f32" ? sizeof(float) : sizeof(double)),
            "bytes,", ap->dtype);

        auto len = ap->len;

        auto m   = static_cast<size_t>(ceil(sqrt(len)));
        auto mxm = m * m;

        LogAll(log_dbg, "add padding:", m, "units");

        auto a = hires::now();
        CHECK_CUDA(cudaMallocHost(&data, mxm * sizeof(DataInUse)));
        memset(data, 0x00, mxm * sizeof(DataInUse));
        io::ReadBinaryToArray<DataInUse>(subfiles.path2file, data, len);
        DataInUse* d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(data, mxm);
        auto       z      = hires::now();

        LogAll(log_dbg, "time loading datum:", static_cast<duration_t>(z - a).count(), "sec");

        datapack.SetHostSpace(data).SetDeviceSpace(d_data).SetLen(len, true);

        if (ap->mode == "r2r") {
            Analyzer analyzer;
            auto     result =
                analyzer.GetMaxMinRng<DataInUse, AnalyzerExecutionPolicy::cuda_device, AnalyzerMethod::thrust>(
                    d_data, len);

            LogAll(log_dbg, "time scanning:", result.seconds, "sec");

            ap->eb *= result.rng;
        }

        LogAll(
            log_dbg, std::to_string(ap->quant_byte) + "-byte quant type,",
            std::to_string(ap->huff_byte) + "-byte internal Huff type");
    }

    if (wf.pre_binning) {
        cerr << log_err
             << "Binning is not working temporarily; we are improving end-to-end throughput by NOT touching "
                "filesystem. (ver. 0.1.4)"
             << endl;
        exit(1);
    }

    if (wf.lossy_construct or wf.lossy_dryrun) {  // fp32 only for now

        if (ap->quant_byte == 1) {
            if (ap->huff_byte == 4)
                cusz::interface::Compress<true, 4, 1, 4>(
                    ap, &datapack, nnz_outlier, total_bits, total_uInt, huff_meta_size, nvcomp_in_use);
            else
                cusz::interface::Compress<true, 4, 1, 8>(
                    ap, &datapack, nnz_outlier, total_bits, total_uInt, huff_meta_size, nvcomp_in_use);
        }
        else if (ap->quant_byte == 2) {
            if (ap->huff_byte == 4)
                cusz::interface::Compress<true, 4, 2, 4>(
                    ap, &datapack, nnz_outlier, total_bits, total_uInt, huff_meta_size, nvcomp_in_use);
            else
                cusz::interface::Compress<true, 4, 2, 8>(
                    ap, &datapack, nnz_outlier, total_bits, total_uInt, huff_meta_size, nvcomp_in_use);
        }

        // pack metadata
        auto mp = new metadata_pack();
        PackMetadata(ap, mp, nnz_outlier);
        mp->total_bits     = total_bits;
        mp->total_uInt     = total_uInt;
        mp->huff_meta_size = huff_meta_size;
        mp->nvcomp_in_use  = nvcomp_in_use;

        auto mp_byte = reinterpret_cast<char*>(mp);
        // yet another metadata package
        io::WriteArrayToBinary(subfiles.compress.out_yamp, mp_byte, sizeof(metadata_pack));

        delete mp;
    }

    if (data) {
        cudaFreeHost(data);  // really messy considering adp pointers are freed elsewhere
        data = nullptr;
    }

    // invoke system() to untar archived files first before decompression
    if (not wf.lossy_construct and wf.lossy_reconstruct) {
        string cx_directory = subfiles.path2file.substr(0, subfiles.path2file.rfind('/') + 1);
        string cmd_string;
        if (cx_directory.length() == 0)
            cmd_string = "tar -xf " + subfiles.path2file + ".sz";
        else
            cmd_string = "tar -xf " + subfiles.path2file + ".sz" + " -C " + cx_directory;

        CheckShellCall(cmd_string);
    }

    if (wf.lossy_reconstruct) {  // fp32 only for now

        // unpack metadata
        auto mp_byte = io::ReadBinaryToNewArray<char>(subfiles.decompress.in_yamp, sizeof(metadata_pack));
        auto mp      = reinterpret_cast<metadata_pack*>(mp_byte);

        UnpackMetadata(ap, mp, nnz_outlier);
        total_bits     = mp->total_bits;
        total_uInt     = mp->total_uInt;
        huff_meta_size = mp->huff_meta_size;
        nvcomp_in_use  = mp->nvcomp_in_use;

        if (ap->quant_byte == 1) {
            if (ap->huff_byte == 4)
                cusz::interface::Decompress<true, 4, 1, 4>(
                    ap, nnz_outlier, total_bits, total_uInt, huff_meta_size, nvcomp_in_use);
            else if (ap->huff_byte == 8)
                cusz::interface::Decompress<true, 4, 1, 8>(
                    ap, nnz_outlier, total_bits, total_uInt, huff_meta_size, nvcomp_in_use);
        }
        else if (ap->quant_byte == 2) {
            if (ap->huff_byte == 4)
                cusz::interface::Decompress<true, 4, 2, 4>(
                    ap, nnz_outlier, total_bits, total_uInt, huff_meta_size, nvcomp_in_use);
            else if (ap->huff_byte == 8)
                cusz::interface::Decompress<true, 4, 2, 8>(
                    ap, nnz_outlier, total_bits, total_uInt, huff_meta_size, nvcomp_in_use);
        }
    }

    // invoke system() function to merge and compress the resulting 5 files after cusz compression
    string basename = subfiles.path2file.substr(subfiles.path2file.rfind('/') + 1);
    if (not wf.lossy_reconstruct and wf.lossy_construct) {
        auto tar_a = hires::now();

        // remove *.sz if existing
        string cmd_string = "rm -rf " + ap->opath + basename + ".sz";
        CheckShellCall(cmd_string);

        // using tar command to encapsulate files
        string files_to_merge;
        if (wf.skip_huffman_enc) {
            files_to_merge = basename + ".outlier " + basename + ".quant " + basename + ".yamp";
        }
        else {
            files_to_merge = basename + ".hbyte " + basename + ".outlier " + basename + ".canon " + basename +
                             ".hmeta " + basename + ".yamp";
        }
        if (wf.lossless_gzip) { cmd_string = "cd " + ap->opath + ";tar -czf " + basename + ".sz " + files_to_merge; }
        else {
            cmd_string = "cd " + ap->opath + ";tar -cf " + basename + ".sz " + files_to_merge;
        }
        CheckShellCall(cmd_string);

        // remove 5 subfiles
        cmd_string = "cd " + ap->opath + ";rm -rf " + files_to_merge;
        CheckShellCall(cmd_string);

        auto tar_z = hires::now();

        auto ad_hoc_fix = ap->opath.substr(0, ap->opath.size() - 1);
        LogAll(log_dbg, "time tar'ing:", static_cast<duration_t>(tar_z - tar_a).count(), "sec");
        LogAll(log_info, "output:", ad_hoc_fix + basename + ".sz");
    }

    // if it's decompression, remove released subfiles at last.
    if (not wf.lossy_construct and wf.lossy_reconstruct) {
        string files_to_delete;
        if (wf.skip_huffman_enc) {
            files_to_delete = basename + ".outlier " + basename + ".quant " + basename + ".yamp";
        }
        else {
            files_to_delete = basename + ".hbyte " + basename + ".outlier " + basename + ".canon " + basename +
                              ".hmeta " + basename + ".yamp";
        }
        string cmd_string =
            "cd " + subfiles.path2file.substr(0, subfiles.path2file.rfind('/')) + ";rm -rf " + files_to_delete;
        CheckShellCall(cmd_string);
    }

    if (wf.lossy_construct and wf.lossy_reconstruct) {
        // remove *.sz if existing
        string cmd_string = "rm -rf " + ap->opath + basename + ".sz";
        CheckShellCall(cmd_string);

        // using tar command to encapsulate files
        string files_for_merging;
        if (wf.skip_huffman_enc) {
            files_for_merging = basename + ".outlier " + basename + ".quant " + basename + ".yamp";
        }
        else {
            files_for_merging = basename + ".hbyte " + basename + ".outlier " + basename + ".canon " + basename +
                                ".hmeta " + basename + ".yamp";
        }
        if (wf.lossless_gzip) { cmd_string = "cd " + ap->opath + ";tar -czf " + basename + ".sz " + files_for_merging; }
        else {
            cmd_string = "cd " + ap->opath + ";tar -cf " + basename + ".sz " + files_for_merging;
        }
        CheckShellCall(cmd_string);

        // remove 5 subfiles
        cmd_string = "cd " + ap->opath + ";rm -rf " + files_for_merging;
        CheckShellCall(cmd_string);

        LogAll(log_info, "write to: " + ap->opath + basename + ".sz");
        LogAll(log_info, "write to: " + ap->opath + basename + ".szx");

        if (wf.gtest) {
            expectedErr  = ap->eb;
            z_mode       = ap->mode;
            auto stat    = ap->stat;
            actualAbsErr = stat.max_abserr;
            actualRelErr = stat.max_abserr_vs_rng;
            ::testing::InitGoogleTest(&argc, argv);
            return RUN_ALL_TESTS();
        }
    }
}

#endif
