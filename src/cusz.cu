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

#include "analysis_utils.hh"
#include "argparse.hh"
#include "cusz_interface.cuh"
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

    auto get_nblk = [&](int d) { return (d + ap->GPU_block_size - 1) / ap->GPU_block_size; };

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
    auto ap = new argpack(argc, argv);

    int    nnz_outlier = 0;
    size_t total_bits, total_uInt, huff_meta_size;
    bool   nvcomp_in_use = false;

    if (ap->verbose) {
        GetMachineProperties();
        GetDeviceProperty();
    }

    // TODO hardcode for float for now
    using DataInUse                  = float;
    struct DataPack<DataInUse>* adp  = nullptr;
    DataInUse*                  data = nullptr;

    if (ap->to_archive or ap->to_dryrun) {
        InitializeDims(ap);

        LogAll(
            log_info, "load", ap->cx_path2file, ap->len * (ap->dtype == "f32" ? sizeof(float) : sizeof(double)),
            "bytes,", ap->dtype);

        auto len = ap->len;

        auto m   = cusz::impl::GetEdgeOfReinterpretedSquare(len);  // row-major mxn matrix
        auto mxm = m * m;

        LogAll(log_dbg, "add padding:", m, "units");

        auto a = hires::now();
        CHECK_CUDA(cudaMallocHost(&data, mxm * sizeof(DataInUse)));
        memset(data, 0x00, mxm * sizeof(DataInUse));
        io::ReadBinaryToArray<DataInUse>(ap->cx_path2file, data, len);
        DataInUse* d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(data, mxm);
        auto       z      = hires::now();

        LogAll(log_dbg, "time loading datum:", static_cast<duration_t>(z - a).count(), "sec");

        adp = new DataPack<DataInUse>(data, d_data, len);

        // TODO parse input directly
        ap->eb = ap->mantissa * std::pow(10, ap->exponent);

        if (ap->mode == "r2r") {
            double rng;
            auto   time_0 = hires::now();
            // TODO move to data analytics
            // ------------------------------------------------------------
            thrust::device_ptr<float> g_ptr = thrust::device_pointer_cast(d_data);

            size_t min_el_loc = thrust::min_element(g_ptr, g_ptr + len) - g_ptr;  // excluding padded
            size_t max_el_loc = thrust::max_element(g_ptr, g_ptr + len) - g_ptr;  // excluding padded

            double min_value = *(g_ptr + min_el_loc);
            double max_value = *(g_ptr + max_el_loc);
            rng              = max_value - min_value;
            // ------------------------------------------------------------
            auto time_1 = hires::now();

            LogAll(log_dbg, "time scanning:", static_cast<duration_t>(time_1 - time_0).count(), "sec");

            ap->eb *= rng;
        }

        LogAll(
            log_dbg, std::to_string(ap->quant_byte) + "-byte quant type,",
            std::to_string(ap->huff_byte) + "-byte internal Huff type");
    }

    if (ap->pre_binning) {
        cerr << log_err
             << "Binning is not working temporarily; we are improving end-to-end throughput by NOT touching "
                "filesystem. (ver. 0.1.4)"
             << endl;
        exit(1);
    }

    if (ap->to_archive or ap->to_dryrun) {  // fp32 only for now

        if (ap->quant_byte == 1) {
            if (ap->huff_byte == 4)
                cusz::interface::Compress<true, 4, 1, 4>(
                    ap, adp, nnz_outlier, total_bits, total_uInt, huff_meta_size, nvcomp_in_use);
            else
                cusz::interface::Compress<true, 4, 1, 8>(
                    ap, adp, nnz_outlier, total_bits, total_uInt, huff_meta_size, nvcomp_in_use);
        }
        else if (ap->quant_byte == 2) {
            if (ap->huff_byte == 4)
                cusz::interface::Compress<true, 4, 2, 4>(
                    ap, adp, nnz_outlier, total_bits, total_uInt, huff_meta_size, nvcomp_in_use);
            else
                cusz::interface::Compress<true, 4, 2, 8>(
                    ap, adp, nnz_outlier, total_bits, total_uInt, huff_meta_size, nvcomp_in_use);
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
        io::WriteArrayToBinary(ap->c_fo_yamp, mp_byte, sizeof(metadata_pack));

        delete mp;
    }

    if (data and adp) {
        cudaFreeHost(data);  // really messy considering adp pointers are freed elsewhere
        data = nullptr;
        delete adp;
    }

    // before running into decompression, reset for `-z -x`
    {
        //        delete ap;
        //        ap = new ArgPack();
    }

    // wenyu's modification
    // invoke system() to untar archived files first before decompression

    if (not ap->to_archive && ap->to_extract) {
        string cx_directory = ap->cx_path2file.substr(0, ap->cx_path2file.rfind('/') + 1);
        string cmd_string;
        if (cx_directory.length() == 0)
            cmd_string = "tar -xf " + ap->cx_path2file + ".sz";
        else
            cmd_string = "tar -xf " + ap->cx_path2file + ".sz" + " -C " + cx_directory;

        CheckShellCall(cmd_string);
    }

    // wenyu's modification ends

    if (ap->to_extract) {  // fp32 only for now

        // unpack metadata
        auto mp_byte = io::ReadBinaryToNewArray<char>(ap->x_fi_yamp, sizeof(metadata_pack));
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

    // wenyu's modification starts
    // invoke system() function to merge and compress the resulting 5 files after cusz compression
    string cx_basename = ap->cx_path2file.substr(ap->cx_path2file.rfind('/') + 1);
    if (not ap->to_extract && ap->to_archive) {
        auto tar_a = hires::now();

        // remove *.sz if existing
        string cmd_string = "rm -rf " + ap->opath + cx_basename + ".sz";
        CheckShellCall(cmd_string);

        // using tar command to encapsulate files
        string files_for_merging;
        if (ap->skip_huffman) {
            files_for_merging = cx_basename + ".outlier " + cx_basename + ".quant " + cx_basename + ".yamp";
        }
        else {
            files_for_merging = cx_basename + ".hbyte " + cx_basename + ".outlier " + cx_basename + ".canon " +
                                cx_basename + ".hmeta " + cx_basename + ".yamp";
        }
        if (ap->to_gzip) { cmd_string = "cd " + ap->opath + ";tar -czf " + cx_basename + ".sz " + files_for_merging; }
        else {
            cmd_string = "cd " + ap->opath + ";tar -cf " + cx_basename + ".sz " + files_for_merging;
        }
        CheckShellCall(cmd_string);

        // remove 5 subfiles
        cmd_string = "cd " + ap->opath + ";rm -rf " + files_for_merging;
        CheckShellCall(cmd_string);

        auto tar_z = hires::now();

        auto ad_hoc_fix = ap->opath.substr(0, ap->opath.size() - 1);
        LogAll(log_dbg, "time tar'ing:", static_cast<duration_t>(tar_z - tar_a).count(), "sec");
        LogAll(log_info, "output:", ad_hoc_fix + cx_basename + ".sz");
    }

    // if it's decompression, remove released subfiles at last.

    if (not ap->to_archive && ap->to_extract) {
        string files_for_deleting;
        if (ap->skip_huffman) {
            files_for_deleting = cx_basename + ".outlier " + cx_basename + ".quant " + cx_basename + ".yamp";
        }
        else {
            files_for_deleting = cx_basename + ".hbyte " + cx_basename + ".outlier " + cx_basename + ".canon " +
                                 cx_basename + ".hmeta " + cx_basename + ".yamp";
        }
        string cmd_string =
            "cd " + ap->cx_path2file.substr(0, ap->cx_path2file.rfind('/')) + ";rm -rf " + files_for_deleting;
        CheckShellCall(cmd_string);
    }

    if (ap->to_archive && ap->to_extract) {
        // remove *.sz if existing
        string cmd_string = "rm -rf " + ap->opath + cx_basename + ".sz";
        CheckShellCall(cmd_string);

        // using tar command to encapsulate files
        string files_for_merging;
        if (ap->skip_huffman) {
            files_for_merging = cx_basename + ".outlier " + cx_basename + ".quant " + cx_basename + ".yamp";
        }
        else {
            files_for_merging = cx_basename + ".hbyte " + cx_basename + ".outlier " + cx_basename + ".canon " +
                                cx_basename + ".hmeta " + cx_basename + ".yamp";
        }
        if (ap->to_gzip) { cmd_string = "cd " + ap->opath + ";tar -czf " + cx_basename + ".sz " + files_for_merging; }
        else {
            cmd_string = "cd " + ap->opath + ";tar -cf " + cx_basename + ".sz " + files_for_merging;
        }
        CheckShellCall(cmd_string);

        // remove 5 subfiles
        cmd_string = "cd " + ap->opath + ";rm -rf " + files_for_merging;
        CheckShellCall(cmd_string);

        LogAll(log_info, "write to: " + ap->opath + cx_basename + ".sz");
        LogAll(log_info, "write to: " + ap->opath + cx_basename + ".szx");

        if (ap->to_gtest) {
            expectedErr  = ap->mantissa * std::pow(10, ap->exponent);
            z_mode       = ap->mode;
            auto stat    = ap->stat;
            actualAbsErr = stat.max_abserr;
            actualRelErr = stat.max_abserr_vs_range;
            ::testing::InitGoogleTest(&argc, argv);
            return RUN_ALL_TESTS();
        }
    }

    // wenyu's modification ends
}

#endif
