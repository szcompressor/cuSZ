/**
 * @file withwrapper_lorenzo.cu
 * @author Jiannan Tian
 * @brief A temporary test case using high-level wrapper/API.
 * @version 0.3
 * @date 2021-06-21
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <iostream>
#include <string>
#include "../utils/io.hh"
#include "../utils/verify.hh"

#pragma message "--extended-lambda causes migration error (nvcc is incapable to be a wellrounded compiler)."
// #include "../utils/verify_gpu.cuh"
#include "../wrapper/extrap_lorenzo.h"

using std::cout;
using std::endl;

using Data  = float;
using Quant = uint16_t;
using FP    = float;

Data eb;
Data maxval, minval;

// dim3   stride3;
size_t len1;
int    radius = 512;

namespace {

#ifndef __CUDACC__
struct __dim3_compat {
    unsigned int x, y, z;
    __dim3_compat(unsigned int _x, unsigned int _y, unsigned int _z){};
};

using dim3 = __dim3_compat;
#endif

auto get_npart = [](auto size, auto subsize) {
    static_assert(
        std::numeric_limits<decltype(size)>::is_integer and std::numeric_limits<decltype(subsize)>::is_integer,
        "[get_npart] must be plain interger types.");
    return (size + subsize - 1) / subsize;
};
auto get_len_from_dim3 = [](dim3 size) { return size.x * size.y * size.z; };
auto get_stride3       = [](dim3 size) -> dim3 { return dim3(1, size.x, size.x * size.y); };

}  // namespace

void test_lorenzo(std::string fname, int ndim, dim3 size3)
{
    cout << "filename: " << fname << '\n';

    Data*  h_data{nullptr};
    Data*  d_data{nullptr};
    Data*  h2_data{nullptr};
    Quant* d_quant{nullptr};

    auto len1 = get_len_from_dim3(size3);
    cout << "len1 from dim3: " << len1 << endl;

    cudaMallocHost(&h_data, len1 * sizeof(Data));
    io::read_binary_to_array(fname, h_data, len1);
    cudaMallocHost(&h2_data, len1 * sizeof(Data));
    memcpy(h2_data, h_data, len1 * sizeof(Data));

    cudaMalloc(&d_data, len1 * sizeof(Data));
    cudaMemcpy(d_data, h_data, len1 * sizeof(Data), cudaMemcpyHostToDevice);
    cudaMalloc(&d_quant, len1 * sizeof(Quant));

    auto maxval = *std::max_element(h_data, h_data + len1);
    auto minval = *std::min_element(h_data, h_data + len1);
    eb          = 1e-3 * (maxval - minval);

    compress_lorenzo_construct<Data, Quant, FP>(d_data, d_quant, size3, ndim, eb, radius);
    decompress_lorenzo_reconstruct<Data, Quant, FP>(d_data, d_quant, size3, ndim, eb, radius);

    cudaMemcpy(h_data, d_data, len1 * sizeof(Data), cudaMemcpyDeviceToHost);

    // TODO GPU verification does not print
    // {
    //     stat_t stat_gpu;
    //     verify_data_GPU(&stat_gpu, h_data, h2_data, len1);
    //     analysis::print_data_quality_metrics<Data>(&stat_gpu, false, eb, 0, 1, false, true);
    // }
    {
        stat_t stat;
        analysis::verify_data(&stat, h_data, h2_data, len1);
        analysis::print_data_quality_metrics<Data>(&stat, false, eb, 0, 1, false, false);
    }

    // clear up
    cudaFree(d_data);
    cudaFree(d_quant);
    cudaFreeHost(h_data);
    cudaFreeHost(h2_data);
}

int main()
{
    struct passwd* pw      = getpwuid(getuid());
    const char*    homedir = pw->pw_dir;

    test_lorenzo(std::string(homedir) + "/datafields/vx", 1, dim3(280953867, 1, 1));
    test_lorenzo(std::string(homedir) + "/datafields/CLDHGH", 2, dim3(3600, 1800, 1));
    test_lorenzo(std::string(homedir) + "/datafields/CLOUDf48", 3, dim3(500, 500, 100));

    return 0;
}
