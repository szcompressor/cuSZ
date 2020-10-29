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

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "SDRB.hh"
#include "argparse.hh"
#include "constants.hh"
#include "cuda_mem.cuh"
#include "cusz_workflow.cuh"
#include "filter.cuh"
#include "io.hh"
#include "pack.hh"
#include "query.hh"
#include "types.hh"

using std::string;

template <typename T, int DS, int tBLK>
T* pre_binning(T* d, size_t* dim_array)
{
    auto d0      = dim_array[DIM0];
    auto d1      = dim_array[DIM1];
    auto d2      = dim_array[DIM2];
    auto d3      = dim_array[DIM3];
    auto len     = d0 * d1 * d2 * d3;
    auto new_d0  = (dim_array[DIM0] - 1) / DS + 1;
    auto new_d1  = (dim_array[DIM1] - 1) / DS + 1;
    auto new_d2  = (dim_array[DIM2] - 1) / DS + 1;
    auto new_d3  = (dim_array[DIM3] - 1) / DS + 1;
    auto new_len = new_d0 * new_d1 * new_d2 * new_d3;

    size_t new_dims[] = {new_d0, new_d1, new_d2, new_d3};
    SetDims(dim_array, new_dims);

    auto d_d  = mem::CreateDeviceSpaceAndMemcpyFromHost(d, len);
    auto d_ds = mem::CreateCUDASpace<T>(new_len);

    dim3 blockDim_binning(tBLK, tBLK);
    dim3 gridDim_binning((new_d0 - 1) / tBLK + 1, (new_d1 - 1) / tBLK + 1);
    Prototype::binning2d<T, DS, tBLK><<<gridDim_binning, blockDim_binning>>>(d_d, d_ds, d0, d1, new_d0, new_d1);
    cudaDeviceSynchronize();

    cudaFree(d_d);
    return d_ds;
}

int main(int argc, char** argv)
{
    auto ap = new argpack(argc, argv);

    size_t* dim_array   = nullptr;
    double* eb_array    = nullptr;
    int     nnz_outlier = 0;
    size_t  total_bits, total_uInt, huff_meta_size;

    if (ap->verbose) {
        GetMachineProperties();
        GetDeviceProperty();
    }

    if (ap->to_archive or ap->to_dryrun) {
        dim_array = ap->use_demo ? InitializeDemoDims(ap->demo_dataset, ap->dict_size)  //
                                 : InitializeDims(ap->dict_size, ap->n_dim, ap->d0, ap->d1, ap->d2, ap->d3);

        cout << log_info;
        printf(
            "datum:\t\t%s (%lu bytes) of type %s\n", ap->cx_path2file.c_str(),
            dim_array[LEN] * (ap->dtype == "f32" ? sizeof(float) : sizeof(double)), ap->dtype.c_str());

        auto eb_config = new config_t(ap->dict_size, ap->mantissa, ap->exponent);
        if (ap->mode == "r2r")
            eb_config->ChangeToRelativeMode(GetDatumValueRange<float>(ap->cx_path2file, dim_array[LEN]));
        // eb_config->debug();
        eb_array = InitializeErrorBoundFamily(eb_config);

        cout << log_dbg << "\e[1m" << ap->quant_rep / 8 << "-byte\e[0m quant type, \e[1m" << ap->huffman_rep / 8
             << "-byte\e[0m internal Huff type" << endl;
    }

    if (ap->pre_binning) {
        auto data        = io::ReadBinaryFile<float>(ap->cx_path2file, dim_array[LEN]);
        auto d_binning   = pre_binning<float, 2, 32>(data, dim_array);
        auto binning     = mem::CreateHostSpaceAndMemcpyFromDevice(d_binning, dim_array[LEN]);
        ap->cx_path2file = ap->cx_path2file + ".BN";
        io::WriteArrayToBinary(ap->cx_path2file, binning, dim_array[LEN]);

        cudaFree(d_binning);
        delete[] data;
        delete[] binning;
    }

    if (ap->to_archive or ap->to_dryrun) {  // fp32 only for now
        if (ap->quant_rep == 8) {
            if (ap->huffman_rep == 32)
                cusz::workflow::Compress<float, uint8_t, uint32_t>(
                    ap, dim_array, eb_array, nnz_outlier, total_bits, total_uInt, huff_meta_size);
            else
                cusz::workflow::Compress<float, uint8_t, uint64_t>(
                    ap, dim_array, eb_array, nnz_outlier, total_bits, total_uInt, huff_meta_size);
        }
        else if (ap->quant_rep == 16) {
            if (ap->huffman_rep == 32)
                cusz::workflow::Compress<float, uint16_t, uint32_t>(
                    ap, dim_array, eb_array, nnz_outlier, total_bits, total_uInt, huff_meta_size);
            else
                cusz::workflow::Compress<float, uint16_t, uint64_t>(
                    ap, dim_array, eb_array, nnz_outlier, total_bits, total_uInt, huff_meta_size);
        }

        // pack metadata
        auto mp = new metadata_pack();
        PackMetadata(ap, mp, nnz_outlier, dim_array, eb_array);
        mp->total_bits     = total_bits;
        mp->total_uInt     = total_uInt;
        mp->huff_meta_size = huff_meta_size;

        auto mp_byte = reinterpret_cast<char*>(mp);
        // yet another metadata package
        io::WriteArrayToBinary(ap->c_fo_yamp, mp_byte, sizeof(metadata_pack));

        delete mp;
    }

    {
        // reset anyway
        // TODO shared pointer
        if (dim_array) memset(dim_array, 0, sizeof(size_t) * 16);
        if (eb_array) memset(eb_array, 0, sizeof(double) * 4);
    }

    // wenyu's modification
    // invoke system() to untar archived files first before decompression
    
    string cx_directory = ap->cx_path2file.substr(0,ap->cx_path2file.rfind("/") + 1);
    if (ap->to_extract) {
        string cmd_string;
	if(cx_directory.length()==0){
	    cmd_string = "tar -xf " + ap->cx_path2file + ".sz";
	} else {
	    cmd_string = "tar -xf " + ap->cx_path2file + ".sz "+" -C "+cx_directory;
	}
        char*  cmd        = new char[cmd_string.length() + 1];
        strcpy(cmd, cmd_string.c_str());
        system(cmd);
        delete[] cmd;
    }

    // wenyu's modification ends

    if (ap->to_extract) {  // fp32 only for now

        // unpack metadata
        auto mp_byte = io::ReadBinaryToNewArray<char>(ap->x_fi_yamp, sizeof(metadata_pack));
        auto mp      = reinterpret_cast<metadata_pack*>(mp_byte);
        if (not dim_array) dim_array = new size_t[16];
        if (not eb_array) eb_array = new double[4];
        UnpackMetadata(ap, mp, nnz_outlier, dim_array, eb_array);
        total_bits     = mp->total_bits;
        total_uInt     = mp->total_uInt;
        huff_meta_size = mp->huff_meta_size;

        if (ap->quant_rep == 8) {
            if (ap->huffman_rep == 32)
                cusz::workflow::Decompress<float, uint8_t, uint32_t>(
                    ap, dim_array, eb_array, nnz_outlier, total_bits, total_uInt, huff_meta_size);
            else
                cusz::workflow::Decompress<float, uint8_t, uint64_t>(
                    ap, dim_array, eb_array, nnz_outlier, total_bits, total_uInt, huff_meta_size);
        }
        else if (ap->quant_rep == 16) {
            if (ap->huffman_rep == 32)
                cusz::workflow::Decompress<float, uint16_t, uint32_t>(
                    ap, dim_array, eb_array, nnz_outlier, total_bits, total_uInt, huff_meta_size);
            else
                cusz::workflow::Decompress<float, uint16_t, uint64_t>(
                    ap, dim_array, eb_array, nnz_outlier, total_bits, total_uInt, huff_meta_size);
        }
    }

    delete[] dim_array;
    delete[] eb_array;
    // delete eb_config;

    // wenyu's modification starts
    // invoke system() function to merge and compress the resulting 5 files after cusz compression
    string cx_basename = ap->cx_path2file.substr(ap->cx_path2file.rfind("/") + 1);
    if (ap->to_archive or ap->to_dryrun) {
        // remove *.sz if existing
        string cmd_string = "rm -rf " + ap->opath + cx_basename + ".sz";
        char*  cmd        = new char[cmd_string.length() + 1];
        strcpy(cmd, cmd_string.c_str());
        system(cmd);
        delete[] cmd;

        // using tar command to encapsulate files
        
        if(ap->to_gzip){
            cmd_string = "cd " + ap->opath + ";tar -czf " + cx_basename + ".sz " + cx_basename + ".hbyte " + cx_basename +
                     ".outlier " + cx_basename + ".canon " + cx_basename + ".hmeta " + cx_basename + ".yamp";
        } else {
            cmd_string = "cd " + ap->opath + ";tar -cf " + cx_basename + ".sz " + cx_basename + ".hbyte " + cx_basename +
                     ".outlier " + cx_basename + ".canon " + cx_basename + ".hmeta " + cx_basename + ".yamp";
        }

        cmd = new char[cmd_string.length() + 1];
        strcpy(cmd, cmd_string.c_str());
        system(cmd);
        delete[] cmd;
        cout << log_info << "Written to:\t\e[1m" << ap->opath << cx_basename << ".sz\e[0m" << endl;

        // remove 5 subfiles
        cmd_string = "rm -rf " + ap->opath + cx_basename + ".hbyte " + ap->opath + cx_basename + ".outlier " +
                     ap->opath + cx_basename + ".canon " + ap->opath + cx_basename + ".hmeta " + ap->opath +
                     cx_basename + ".yamp";
        cmd = new char[cmd_string.length() + 1];
        strcpy(cmd, cmd_string.c_str());
        system(cmd);
        delete[] cmd;
    }

    // if it's decompression, remove released subfiles at last.

    if (ap->to_extract) {
        string cmd_string = "rm -rf " + ap->cx_path2file + ".hbyte " + ap->cx_path2file + ".outlier " +
                            ap->cx_path2file + ".canon " + ap->cx_path2file + ".hmeta " + ap->cx_path2file + ".yamp";
        char* cmd = new char[cmd_string.length() + 1];
        strcpy(cmd, cmd_string.c_str());
        system(cmd);
        delete[] cmd;
    }

    // wenyu's modification ends
}
