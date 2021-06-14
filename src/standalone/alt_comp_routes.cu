/**
 * @file run_rle.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-04-02
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <iostream>
#include <string>

#include "../analysis/analyzer.hh"
#include "../kernel/lorenzo.h"
#include "../ood/codec_runlength.hh"
#include "../utils/io.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

// data types
using Data  = float;
using Quant = unsigned short;
using Count = int;

// global variables
auto     radius = 512;
Analyzer analyzer;
string   fname;
float    eb = 1e-2;

Data*  data;
Quant* quant;
void*  void_rle_keys;
void*  void_error_control;
Count* rle_lens;

auto num_partitions = [](auto size, auto subsize) { return (size + subsize - 1) / subsize; };

int dimx, dimy, dimz, stridey, stridez;
int len;

auto lorenzo_dim_block = dim3(32, 1, 8);
auto lorenzo_dim_grid  = dim3(8, 57, 57);

template <typename Key = int>
void clorenzo_alternative_rle()
{
    // lorenzo
    auto error_control = static_cast<Key*>(void_error_control);
    cudaMallocManaged(&error_control, len * sizeof(Key));

    // RLE
    auto rle_keys = static_cast<Key*>(void_rle_keys);

    cudaMallocManaged(&rle_keys, len * sizeof(Quant));
    cudaMallocManaged(&rle_lens, len * sizeof(Count));

    auto ebx2_r = 1 / (eb * 2);

    for (auto i = 0; i < 10; i++) {
        kernel::c_lorenzo_3d1l_v1_32x8x8data_mapto_32x1x8<Data, Quant, float, true, Key>  //
            <<<lorenzo_dim_grid, lorenzo_dim_block>>>(
                data, quant, dimx, dimy, dimz, stridey, stridez, radius, ebx2_r, error_control);
        cudaDeviceSynchronize();
    }

    // cout << "analyzing error_control" << endl;
    // auto res1 = analyzer.GetMaxMinRng                                                  //
    //             <Count, AnalyzerExecutionPolicy::cuda_device, AnalyzerMethod::thrust>  //
    //             (error_control, len);
    // cout << '\n\n';

    auto codec    = RunLengthCodecAdHoc<Key>(error_control, rle_keys, rle_lens);
    auto num_runs = codec.SetFullLen(len).Encode().RunLen();

    cout << "num runs:\t" << num_runs << endl;

    cout << "analyzing rle_lens" << endl;
    auto res2 = analyzer.GetMaxMinRng                                                  //
                <Count, AnalyzerExecutionPolicy::cuda_device, AnalyzerMethod::thrust>  //
                (rle_lens, len);

    cout << "file:\t" << fname << '\n';
    cout << "max run-length:\t" << res2.max_val << '\n';

    // statistics
    auto ori_len  = sizeof(float) * len;  // original data is float32
    auto compact1 = (sizeof(Key) + sizeof(uint32_t)) * num_runs;
    auto compact2 = (sizeof(Key) + sizeof(uint16_t)) * num_runs;
    auto compact3 = (sizeof(Key) + sizeof(uint8_t)) * num_runs;

    auto h_freq = new unsigned int[256]();
    auto h_c1   = reinterpret_cast<uint8_t*>(rle_lens);
    for (auto i = 0; i < num_runs * sizeof(Count); i++) { h_freq[h_c1[i]] += 1; }

    auto h_c2 = reinterpret_cast<uint8_t*>(rle_keys);
    for (auto i = 0; i < num_runs * sizeof(Key); i++) { h_freq[h_c2[i]] += 1; }

    analyzer.EstimateFromHistogram(h_freq, 256).PrintCompressibilityInfo(false, false, 8);

    cout << "reduction rate (uint32_t):\t" << (1.0 * ori_len) / compact1 << '\n';
    cout << "reduction rate (uint16, if possible):\t" << (1.0 * ori_len) / compact2 << '\n';
    cout << "reduction rate (uint8, if possible):\t" << (1.0 * ori_len) / compact3 << '\n';
    cout << endl;

    cudaFree(error_control);
    cudaFree(rle_keys);
    cudaFree(rle_lens);
}

int main(int argc, char** argv)
{
    if (argc != 5)  //
    {
        cout << "help:\n  altrle  <path/to/input> <x> <y> <z>\n\n";
        exit(0);
    }

    {
        fname = string(argv[1]);
        dimx  = std::atoi(argv[2]);
        dimy  = std::atoi(argv[3]);
        dimz  = std::atoi(argv[4]);

        stridey = dimx;
        stridez = dimx * dimy;
        len     = dimx * dimy * dimz;
    }

    cudaMallocManaged(&data, len * sizeof(Data));
    io::ReadBinaryToArray(fname, data, len);
    auto res = analyzer.GetMaxMinRng                                                 //
               <Data, AnalyzerExecutionPolicy::cuda_device, AnalyzerMethod::thrust>  //
               (data, len);
    eb *= res.rng;

    clorenzo_alternative_rle();
}