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
#include <iomanip>
#include <iostream>
#include <string>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "../utils/io.hh"
#include "../utils/timer.hh"
#include "../utils/verify.hh"

// #pragma message "--extended-lambda causes migration error (nvcc is incapable to be a wellrounded compiler)."
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

#ifndef SYCL_LANGUAGE_VERSION
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
auto get_len_from_dim3 = [](sycl::range<3> size) { return size[2] * size[1] * size[0]; };
auto get_stride3 = [](sycl::range<3> size) -> sycl::range<3> { return sycl::range<3>(size[2] * size[1], size[2], 1); };

}  // namespace

#ifndef DPCPP_QUERY

template <int NDIM, int DATA_SUBSIZE>
void test_lorenzo(std::string fname, sycl::range<3> size3)
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue&      q_ct1   = dev_ct1.default_queue();
    cout << "filename: " << fname << '\n';

    Data*  h_data{nullptr};
    Data*  d_data{nullptr};
    Data*  h2_data{nullptr};
    Quant* d_quant{nullptr};

    auto len1 = get_len_from_dim3(size3);
    cout << "len1 from dim3: " << len1 << endl;

    h_data = sycl::malloc_host<Data>(len1, q_ct1);
    io::ReadBinaryToArray(fname, h_data, len1);
    h2_data = sycl::malloc_host<Data>(len1, q_ct1);
    memcpy(h2_data, h_data, len1 * sizeof(Data));

    d_data = sycl::malloc_device<Data>(len1, q_ct1);
    q_ct1.memcpy(d_data, h_data, len1 * sizeof(Data)).wait();
    d_quant = sycl::malloc_device<Quant>(len1, q_ct1);

    auto maxval = *std::max_element(h_data, h_data + len1);
    auto minval = *std::min_element(h_data, h_data + len1);
    eb          = 1e-3 * (maxval - minval);

    auto tc0 = hires::now();
    compress_lorenzo_construct<Data, Quant, FP, NDIM, DATA_SUBSIZE>(d_data, d_quant, size3, eb, radius);
    auto tc1 = hires::now();
    auto tc  = static_cast<duration_t>(tc1 - tc0).count();
    std::cout << "comp: " << tc << "sec\n";
    std::cout << "comp: " << len1 * 4 / 1024.0 / 1024.0 / 1024.0 / tc << "GB/s\n";

    auto tx0 = hires::now();
    decompress_lorenzo_reconstruct<Data, Quant, FP, NDIM, DATA_SUBSIZE>(d_data, d_quant, size3, eb, radius);
    auto tx1 = hires::now();
    auto tx  = static_cast<duration_t>(tx1 - tx0).count();
    std::cout << "decomp: " << tx << "sec\n";
    std::cout << "decomp: " << len1 * 4 / 1024.0 / 1024.0 / 1024.0 / tx << "GB/\n";

    q_ct1.memcpy(h_data, d_data, len1 * sizeof(Data)).wait();

    // TODO GPU verification does not print
    // {
    //     stat_t stat_gpu;
    //     VerifyDataGPU(&stat_gpu, h_data, h2_data, len1);
    //     analysis::PrintMetrics<Data>(&stat_gpu, false, eb, 0, 1, false, true);
    // }
    {
        stat_t stat;
        analysis::VerifyData(&stat, h_data, h2_data, len1);
        analysis::PrintMetrics<Data>(&stat, false, eb, 0, 1, false, false);
    }

    // clear up
    sycl::free(d_data, q_ct1);
    sycl::free(d_quant, q_ct1);
    sycl::free(h_data, q_ct1);
    sycl::free(h2_data, q_ct1);
}
#endif

#define STRINGFY(X) #X
#define QUERY(ITEM)                                                                                                  \
    std::cout << std::setw(30) << std::left << STRINGFY(ITEM) << "  " << device.get_info<sycl::info::device::ITEM>() \
              << '\n';
#define QUERY_CAST(ITEM, TYPE)                                        \
    std::cout << std::setw(30) << std::left << STRINGFY(ITEM) << "  " \
              << static_cast<TYPE>(device.get_info<sycl::info::device::ITEM>()) << '\n';

#define QUERY_ACCESS(ITEM, IDX)                                                                                        \
    std::cout << std::left << STRINGFY(ITEM) << "." << IDX << "  " << device.get_info<sycl::info::device::ITEM>()[IDX] \
              << '\n';

void query(sycl::info::device_type type)
{
    for (auto device : sycl::device::get_devices(type)) {
        QUERY(name)
        // QUERY(device_type)
        // QUERY(vendor_id)
        QUERY_CAST(max_compute_units, unsigned int)
        QUERY_CAST(max_work_item_dimensions, unsigned int)
        QUERY_ACCESS(max_work_item_sizes, 0)
        QUERY_ACCESS(max_work_item_sizes, 1)
        QUERY_ACCESS(max_work_item_sizes, 2)
        QUERY(max_work_group_size)

        std::cout << '\n';

        QUERY(max_clock_frequency)
        QUERY(max_mem_alloc_size)

        QUERY(global_mem_cache_line_size)
        QUERY(global_mem_cache_size)
        QUERY(global_mem_size)
        QUERY(max_constant_buffer_size)
        // QUERY(max_constant_args)
        // QUERY(local_mem_type)
        QUERY(local_mem_size)
    }
}

int main()
{
#ifdef DPCPP_SHOWCASE
    struct passwd* pw      = getpwuid(getuid());
    const char*    homedir = pw->pw_dir;

    // dpct::device_ext& dev_ct1 = dpct::get_current_device();
    test_lorenzo<1, 256>(std::string(homedir) + "/datafields/vx", sycl::range<3>(1, 1, 280953867));
    test_lorenzo<2, 16>(std::string(homedir) + "/datafields/CLDHGH", sycl::range<3>(1, 1800, 3600));
    test_lorenzo<3, 8>(std::string(homedir) + "/datafields/CLOUDf48", sycl::range<3>(100, 500, 500));
#else

    query(sycl::info::device_type::gpu);
    std::cout << "\n\n";
    query(sycl::info::device_type::cpu);
#endif

    return 0;
}
