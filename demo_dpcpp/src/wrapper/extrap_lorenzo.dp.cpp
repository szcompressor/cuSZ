/**
 * @file extrap_lorenzo.cu
 * @author Jiannan Tian
 * @brief A high-level LorenzoND wrapper. Allocations are explicitly out of called functions.
 * @version 0.3
 * @date 2021-06-16
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "extrap_lorenzo.h"
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "../utils/timer.hh"

#ifdef DPCPP_SHOWCASE
#include "../kernel/lorenzo_prototype.h"

using cusz::prototype::c_lorenzo_1d1l;
using cusz::prototype::c_lorenzo_2d1l;
using cusz::prototype::c_lorenzo_3d1l;
using cusz::prototype::x_lorenzo_1d1l;
using cusz::prototype::x_lorenzo_2d1l;
using cusz::prototype::x_lorenzo_3d1l;

#else
#include "../kernel/lorenzo.h"
#endif

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

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

template <typename Data, typename Quant, typename FP, int NDIM, int DATA_SUBSIZE>
void compress_lorenzo_construct(Data* data, Quant* quant, sycl::range<3> size3, FP eb, int radius)
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue&      q_ct1   = dev_ct1.default_queue();
    FP                ebx2_r  = 1 / (eb * 2);
    auto              stride3 = get_stride3(size3);

    std::cout << "DPCPP compression showcase, revert to prototype ND kernel(s)\n";

    if CONSTEXPR (NDIM == 1) {
        // constexpr auto DATA_SUBSIZE = 256;
        auto dim_block = DATA_SUBSIZE;
        auto dim_grid  = get_npart(size3[2], DATA_SUBSIZE);
        /*
        DPCT1049:23: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler& cgh) {
            sycl::accessor<Data, 1, sycl::access::mode::read_write, sycl::access::target::local> shmem_acc_ct1(
                sycl::range<1>(DATA_SUBSIZE), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, dim_grid) * sycl::range<3>(1, 1, dim_block), sycl::range<3>(1, 1, dim_block)),
                [=](sycl::nd_item<3> item_ct1) {
                    c_lorenzo_1d1l<Data, Quant, FP, DATA_SUBSIZE>(
                        data, quant, size3[2], radius, ebx2_r, item_ct1, shmem_acc_ct1.get_pointer());
                });
        });
    }
    else if CONSTEXPR (NDIM == 2) {
        // constexpr auto DATA_SUBSIZE = 16;
        auto dim_block = sycl::range<3>(1, DATA_SUBSIZE, DATA_SUBSIZE);
        auto dim_grid  = sycl::range<3>(1, get_npart(size3[1], DATA_SUBSIZE), get_npart(size3[2], DATA_SUBSIZE));
        /*
        DPCT1049:24: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler& cgh) {
            sycl::range<2> shmem_range_ct1(DATA_SUBSIZE, DATA_SUBSIZE);

            sycl::accessor<Data, 2, sycl::access::mode::read_write, sycl::access::target::local> shmem_acc_ct1(
                shmem_range_ct1, cgh);

            cgh.parallel_for(sycl::nd_range<3>(dim_grid * dim_block, dim_block), [=](sycl::nd_item<3> item_ct1) {
                c_lorenzo_2d1l<Data, Quant, FP, DATA_SUBSIZE>(
                    data, quant, size3[2], size3[1], size3[2], radius, ebx2_r, item_ct1,
                    dpct::accessor<Data, dpct::local, 2>(shmem_acc_ct1, shmem_range_ct1));
            });
        });
    }
    else if CONSTEXPR (NDIM == 3) {
        // constexpr auto DATA_SUBSIZE = 8;
        auto dim_block = sycl::range<3>(DATA_SUBSIZE, DATA_SUBSIZE, DATA_SUBSIZE);
        auto dim_grid  = sycl::range<3>(
            get_npart(size3[0], DATA_SUBSIZE), get_npart(size3[1], DATA_SUBSIZE), get_npart(size3[2], DATA_SUBSIZE));
        /*
        DPCT1049:25: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler& cgh) {
            sycl::range<3> shmem_range_ct1(DATA_SUBSIZE, DATA_SUBSIZE, DATA_SUBSIZE);

            sycl::accessor<Data, 3, sycl::access::mode::read_write, sycl::access::target::local> shmem_acc_ct1(
                shmem_range_ct1, cgh);

            cgh.parallel_for(sycl::nd_range<3>(dim_grid * dim_block, dim_block), [=](sycl::nd_item<3> item_ct1) {
                c_lorenzo_3d1l<Data, Quant, FP, DATA_SUBSIZE>(
                    data, quant, size3[2], size3[1], size3[0], size3[2], size3[2] * size3[1], radius, ebx2_r, item_ct1,
                    dpct::accessor<Data, dpct::local, 3>(shmem_acc_ct1, shmem_range_ct1));
            });
        });
    }

    dev_ct1.queues_wait_and_throw();
}

/********************************************************************************
 * decompression
 ********************************************************************************/

template <typename Data, typename Quant, typename FP, int NDIM, int DATA_SUBSIZE>
void decompress_lorenzo_reconstruct(Data* xdata, Quant* quant, sycl::range<3> size3, FP eb, int radius)
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue&      q_ct1   = dev_ct1.default_queue();
    auto              stride3 = get_stride3(size3);
    auto              ebx2    = eb * 2;

    std::cout << "DPCPP decompression showcase, revert to prototype ND kernel(s)\n";
    if CONSTEXPR (NDIM == 1) {  // y-sequentiality == 8
        // constexpr auto DATA_SUBSIZE = 256;
        auto dim_block = DATA_SUBSIZE;
        auto dim_grid  = get_npart(size3[2], DATA_SUBSIZE);
        /*
        DPCT1049:26: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler& cgh) {
            sycl::accessor<Data, 1, sycl::access::mode::read_write, sycl::access::target::local> shmem_acc_ct1(
                sycl::range<1>(DATA_SUBSIZE), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, dim_grid) * sycl::range<3>(1, 1, dim_block), sycl::range<3>(1, 1, dim_block)),
                [=](sycl::nd_item<3> item_ct1) {
                    x_lorenzo_1d1l<Data, Quant, FP, DATA_SUBSIZE>(
                        xdata, quant, size3[2], radius, ebx2, item_ct1, shmem_acc_ct1.get_pointer());
                });
        });
    }
    else if CONSTEXPR (NDIM == 2) {
        // constexpr auto DATA_SUBSIZE = 16;
        auto dim_block = sycl::range<3>(1, DATA_SUBSIZE, DATA_SUBSIZE);
        auto dim_grid  = sycl::range<3>(1, get_npart(size3[1], DATA_SUBSIZE), get_npart(size3[2], DATA_SUBSIZE));
        /*
        DPCT1049:27: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler& cgh) {
            sycl::range<2> shmem_range_ct1(DATA_SUBSIZE, DATA_SUBSIZE);

            sycl::accessor<Data, 2, sycl::access::mode::read_write, sycl::access::target::local> shmem_acc_ct1(
                shmem_range_ct1, cgh);

            cgh.parallel_for(sycl::nd_range<3>(dim_grid * dim_block, dim_block), [=](sycl::nd_item<3> item_ct1) {
                x_lorenzo_2d1l<Data, Quant, FP, DATA_SUBSIZE>(
                    xdata, quant, size3[2], size3[1], size3[2], radius, ebx2, item_ct1,
                    dpct::accessor<Data, dpct::local, 2>(shmem_acc_ct1, shmem_range_ct1));
            });
        });
    }
    else if CONSTEXPR (NDIM == 3) {
        // constexpr auto DATA_SUBSIZE = 8;
        auto dim_block = sycl::range<3>(DATA_SUBSIZE, DATA_SUBSIZE, DATA_SUBSIZE);
        auto dim_grid  = sycl::range<3>(
            get_npart(size3[0], DATA_SUBSIZE), get_npart(size3[1], DATA_SUBSIZE), get_npart(size3[2], DATA_SUBSIZE));
        /*
        DPCT1049:28: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler& cgh) {
            sycl::range<3> shmem_range_ct1(DATA_SUBSIZE, DATA_SUBSIZE, DATA_SUBSIZE);

            sycl::accessor<Data, 3, sycl::access::mode::read_write, sycl::access::target::local> shmem_acc_ct1(
                shmem_range_ct1, cgh);

            cgh.parallel_for(sycl::nd_range<3>(dim_grid * dim_block, dim_block), [=](sycl::nd_item<3> item_ct1) {
                x_lorenzo_3d1l<Data, Quant, FP, DATA_SUBSIZE>(
                    xdata, quant, size3[2], size3[1], size3[0], size3[2], size3[2] * size3[1], radius, ebx2, item_ct1,
                    dpct::accessor<Data, dpct::local, 3>(shmem_acc_ct1, shmem_range_ct1));
            });
        });
    }

    dev_ct1.queues_wait_and_throw();
}

/* instantiate */
#define INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(Data, Quant, FP, NDIM, DATA_SUBSIZE) \
    template void compress_lorenzo_construct<Data, Quant, FP, NDIM, DATA_SUBSIZE>(  \
        Data*, Quant*, sycl::range<3>, FP, int);

INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(float, uint16_t, float, 1, 256)
INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(float, uint16_t, float, 2, 16)
INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(float, uint16_t, float, 3, 8)

#define INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(Data, Quant, FP, NDIM, DATA_SUBSIZE) \
    template void decompress_lorenzo_reconstruct<Data, Quant, FP, NDIM, DATA_SUBSIZE>(  \
        Data*, Quant*, sycl::range<3>, FP, int);

INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(float, uint16_t, float, 1, 256)
INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(float, uint16_t, float, 2, 16)
INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(float, uint16_t, float, 3, 8)
