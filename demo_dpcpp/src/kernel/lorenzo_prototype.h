/**
 * @file lorenzo_prototype.h
 * @author Jiannan Tian
 * @brief (prototype) Dual-Quant Lorenzo method.
 * @version 0.2
 * @date 2021-01-16
 * (create) 2019-09-23; (release) 2020-09-20; (rev1) 2021-01-16; (rev2) 2021-02-20; (rev3) 2021-04-11
 * (rev4) 2021-04-30
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_LORENZO_PROTOTYPE_H
#define CUSZ_LORENZO_PROTOTYPE_H

#include <CL/sycl.hpp>
#include <cstddef>
#include <dpct/dpct.hpp>

// TODO disabling dynamic shmem alloction results in wrong number
// extern __shared__ char scratch[];

using DIM    = unsigned int;
using STRIDE = unsigned int;

namespace cusz {
namespace prototype {  // easy algorithmic description

// clang-format off
template <typename Data, typename Quant, typename FP, int BLOCK = 256> void c_lorenzo_1d1l
(Data*, Quant*, DIM, int, FP, sycl::nd_item<3> item_ct1, Data *shmem);
template <typename Data, typename Quant, typename FP, int BLOCK = 256> void x_lorenzo_1d1l
(Data*, Quant*, DIM, int, FP, sycl::nd_item<3> item_ct1, Data *shmem);

template <typename Data, typename Quant, typename FP, int BLOCK = 16> void c_lorenzo_2d1l
(Data*, Quant*, DIM, DIM, STRIDE, int, FP, sycl::nd_item<3> item_ct1, dpct::accessor<Data, dpct::local, 2> shmem);
template <typename Data, typename Quant, typename FP, int BLOCK = 16> void x_lorenzo_2d1l
(Data*, Quant*, DIM, DIM, STRIDE, int, FP, sycl::nd_item<3> item_ct1, dpct::accessor<Data, dpct::local, 2> shmem);

template <typename Data, typename Quant, typename FP, int BLOCK = 8> void c_lorenzo_3d1l
(Data*, Quant*, DIM, DIM, DIM, STRIDE, STRIDE, int, FP, sycl::nd_item<3> item_ct1, dpct::accessor<Data, dpct::local, 3> shmem);
template <typename Data, typename Quant, typename FP, int BLOCK = 8> void x_lorenzo_3d1l
(Data*, Quant*, DIM, DIM, DIM, STRIDE, STRIDE, int, FP, sycl::nd_item<3> item_ct1, dpct::accessor<Data, dpct::local, 3> shmem);
// clang-format on

}  // namespace prototype
}  // namespace cusz

template <typename Data, typename Quant, typename FP, int BLOCK>
void cusz::prototype::c_lorenzo_1d1l(  //
    Data*            data,
    Quant*           quant,
    DIM              dimx,
    int              radius,
    FP               ebx2_r,
    sycl::nd_item<3> item_ct1,
    Data*            shmem)
{
    auto id = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
    if (id < dimx) {
        shmem[item_ct1.get_local_id(2)] = sycl::round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    item_ct1.barrier();  // global mem accessed

    Data delta =
        shmem[item_ct1.get_local_id(2)] - (item_ct1.get_local_id(2) == 0 ? 0 : shmem[item_ct1.get_local_id(2) - 1]);

    {
        bool quantizable = sycl::fabs(delta) < radius;
        Data candidate   = delta + radius;
        if (id < dimx) {                                // postquant
            data[id]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
            quant[id] = quantizable * static_cast<Quant>(candidate);
        }
    }
    // EOF
}

template <typename Data, typename Quant, typename FP, int BLOCK>
void cusz::prototype::c_lorenzo_2d1l(  //
    Data*                                data,
    Quant*                               quant,
    DIM                                  dimx,
    DIM                                  dimy,
    STRIDE                               stridey,
    int                                  radius,
    FP                                   ebx2_r,
    sycl::nd_item<3>                     item_ct1,
    dpct::accessor<Data, dpct::local, 2> shmem)
{
    auto y = item_ct1.get_local_id(1), x = item_ct1.get_local_id(2);
    auto giy = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + y,
         gix = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + x;

    auto id = gix + giy * stridey;  // low to high dim, inner to outer
    if (gix < dimx and giy < dimy) {
        shmem[y][x] = sycl::round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    item_ct1.barrier();  // global mem accesseds

    Data delta = shmem[y][x] - ((x > 0 ? shmem[y][x - 1] : 0) +                // dist=1
                                (y > 0 ? shmem[y - 1][x] : 0) -                // dist=1
                                (x > 0 and y > 0 ? shmem[y - 1][x - 1] : 0));  // dist=2

    {
        bool quantizable = sycl::fabs(delta) < radius;
        Data candidate   = delta + radius;
        if (gix < dimx and giy < dimy) {
            data[id]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
            quant[id] = quantizable * static_cast<Quant>(candidate);
        }
    }
    // EOF
}

template <typename Data, typename Quant, typename FP, int BLOCK>
void cusz::prototype::c_lorenzo_3d1l(  //
    Data*                                data,
    Quant*                               quant,
    DIM                                  dimx,
    DIM                                  dimy,
    DIM                                  dimz,
    STRIDE                               stridey,
    STRIDE                               stridez,
    int                                  radius,
    FP                                   ebx2_r,
    sycl::nd_item<3>                     item_ct1,
    dpct::accessor<Data, dpct::local, 3> shmem)
{
    auto z = item_ct1.get_local_id(0), y = item_ct1.get_local_id(1), x = item_ct1.get_local_id(2);
    auto giz = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + z,
         giy = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + y,
         gix = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + x;

    auto id = gix + giy * stridey + giz * stridez;  // low to high in dim, inner to outer
    if (gix < dimx and giy < dimy and giz < dimz) {
        shmem[z][y][x] = sycl::round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    item_ct1.barrier();  // global mem accesseds

    Data delta = shmem[z][y][x] - ((z > 0 and y > 0 and x > 0 ? shmem[z - 1][y - 1][x - 1] : 0)  // dist=3
                                   - (y > 0 and x > 0 ? shmem[z][y - 1][x - 1] : 0)              // dist=2
                                   - (z > 0 and x > 0 ? shmem[z - 1][y][x - 1] : 0)              //
                                   - (z > 0 and y > 0 ? shmem[z - 1][y - 1][x] : 0)              //
                                   + (x > 0 ? shmem[z][y][x - 1] : 0)                            // dist=1
                                   + (y > 0 ? shmem[z][y - 1][x] : 0)                            //
                                   + (z > 0 ? shmem[z - 1][y][x] : 0));                          //

    {
        bool quantizable = sycl::fabs(delta) < radius;
        Data candidate   = delta + radius;
        if (gix < dimx and giy < dimy and giz < dimz) {
            data[id]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
            quant[id] = quantizable * static_cast<Quant>(candidate);
        }
    }
    // EOF
}

template <typename Data, typename Quant, typename FP, int BLOCK>
void cusz::prototype::x_lorenzo_1d1l(  //
    Data*            xdata_outlier,
    Quant*           quant,
    DIM              dimx,
    int              radius,
    FP               ebx2,
    sycl::nd_item<3> item_ct1,
    Data*            shmem)
{
    auto id = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);

    if (id < dimx)
        shmem[item_ct1.get_local_id(2)] = xdata_outlier[id] + static_cast<Data>(quant[id]) - radius;  // fuse
    else
        shmem[item_ct1.get_local_id(2)] = 0;
    item_ct1.barrier();  // global mem accessed

    for (auto d = 1; d < BLOCK; d *= 2) {
        Data n = 0;
        if (item_ct1.get_local_id(2) >= d)
            n = shmem[item_ct1.get_local_id(2) - d];  // like __shfl_up_sync(0x1f, var, d); warp_sync
        item_ct1.barrier(sycl::access::fence_space::local_space);

        if (item_ct1.get_local_id(2) >= d) shmem[item_ct1.get_local_id(2)] += n;
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if (id < dimx) { xdata_outlier[id] = shmem[item_ct1.get_local_id(2)] * ebx2; }
}

template <typename Data, typename Quant, typename FP, int BLOCK>
void cusz::prototype::x_lorenzo_2d1l(  //
    Data*                                xdata_outlier,
    Quant*                               quant,
    DIM                                  dimx,
    DIM                                  dimy,
    STRIDE                               stridey,
    int                                  radius,
    FP                                   ebx2,
    sycl::nd_item<3>                     item_ct1,
    dpct::accessor<Data, dpct::local, 2> shmem)
{
    auto giy  = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) + item_ct1.get_local_id(1),
         gix  = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
    size_t id = gix + giy * stridey;

    if (gix < dimx and giy < dimy)
        shmem[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] =
            xdata_outlier[id] + static_cast<Data>(quant[id]) - radius;  // fuse
    else
        shmem[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] = 0;
    item_ct1.barrier();  // global mem accessed

    for (auto d = 1; d < BLOCK; d *= 2) {
        Data n = 0;
        if (item_ct1.get_local_id(2) >= d) n = shmem[item_ct1.get_local_id(1)][item_ct1.get_local_id(2) - d];
        item_ct1.barrier(sycl::access::fence_space::local_space);

        if (item_ct1.get_local_id(2) >= d) shmem[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] += n;
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    for (auto d = 1; d < BLOCK; d *= 2) {
        Data n = 0;
        if (item_ct1.get_local_id(1) >= d) n = shmem[item_ct1.get_local_id(1) - d][item_ct1.get_local_id(2)];
        item_ct1.barrier(sycl::access::fence_space::local_space);

        if (item_ct1.get_local_id(1) >= d) shmem[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] += n;
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if (gix < dimx and giy < dimy) {
        xdata_outlier[id] = shmem[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] * ebx2;
    }
}

template <typename Data, typename Quant, typename FP, int BLOCK>
void cusz::prototype::x_lorenzo_3d1l(  //
    Data*                                xdata_outlier,
    Quant*                               quant,
    DIM                                  dimx,
    DIM                                  dimy,
    DIM                                  dimz,
    STRIDE                               stridey,
    STRIDE                               stridez,
    int                                  radius,
    FP                                   ebx2,
    sycl::nd_item<3>                     item_ct1,
    dpct::accessor<Data, dpct::local, 3> shmem)
{
    auto giz  = item_ct1.get_group(0) * BLOCK + item_ct1.get_local_id(0),
         giy  = item_ct1.get_group(1) * BLOCK + item_ct1.get_local_id(1),
         gix  = item_ct1.get_group(2) * BLOCK + item_ct1.get_local_id(2);
    size_t id = gix + giy * stridey + giz * stridez;  // low to high in dim, inner to outer

    if (gix < dimx and giy < dimy and giz < dimz)
        shmem[item_ct1.get_local_id(0)][item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] =
            xdata_outlier[id] + static_cast<Data>(quant[id]) - radius;  // id
    else
        shmem[item_ct1.get_local_id(0)][item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] = 0;
    item_ct1.barrier();

    for (auto dist = 1; dist < BLOCK; dist *= 2) {
        Data addend = 0;
        if (item_ct1.get_local_id(2) >= dist)
            addend = shmem[item_ct1.get_local_id(0)][item_ct1.get_local_id(1)][item_ct1.get_local_id(2) - dist];
        item_ct1.barrier(sycl::access::fence_space::local_space);

        if (item_ct1.get_local_id(2) >= dist)
            shmem[item_ct1.get_local_id(0)][item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] += addend;
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    for (auto dist = 1; dist < BLOCK; dist *= 2) {
        Data addend = 0;
        if (item_ct1.get_local_id(1) >= dist)
            addend = shmem[item_ct1.get_local_id(0)][item_ct1.get_local_id(1) - dist][item_ct1.get_local_id(2)];
        item_ct1.barrier(sycl::access::fence_space::local_space);

        if (item_ct1.get_local_id(1) >= dist)
            shmem[item_ct1.get_local_id(0)][item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] += addend;
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    for (auto dist = 1; dist < BLOCK; dist *= 2) {
        Data addend = 0;
        if (item_ct1.get_local_id(0) >= dist)
            addend = shmem[item_ct1.get_local_id(0) - dist][item_ct1.get_local_id(1)][item_ct1.get_local_id(2)];
        item_ct1.barrier();

        if (item_ct1.get_local_id(0) >= dist)
            shmem[item_ct1.get_local_id(0)][item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] += addend;
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if (gix < dimx and giy < dimy and giz < dimz) {
        xdata_outlier[id] = shmem[item_ct1.get_local_id(0)][item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] * ebx2;
    }
}

#endif
