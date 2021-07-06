/**
 * @file interp_spline.cu
 * @author Jiannan Tian
 * @brief A high-level Spline3D wrapper. Allocations are explicitly out of called functions.
 * @version 0.3
 * @date 2021-06-15
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <numeric>
#include "interp_spline.h"

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
auto get_nblock3       = [](dim3 size) -> dim3 {
    auto x = get_npart(size.x, 32);
    auto y = get_npart(size.y, 8);
    auto z = get_npart(size.z, 8);
    return dim3(x, y, z);
};
auto get_padded3 = [](dim3 size, dim3 nblk3) -> dim3 {
    auto x = nblk3.x * 32;  // e.g. 235 -> 256
    auto y = nblk3.y * 8;   // e.g. 449 -> 456
    auto z = nblk3.z * 8;   // e.g. 449 -> 456
    return dim3(x, y, z);
};

auto get_nanchor3(dim3 size) -> dim3
{
    auto inrange_nx = int(size.x / 8);
    auto inrange_ny = int(size.y / 8);
    auto inrange_nz = int(size.z / 8);
    auto nx         = inrange_nx + 1;
    auto ny         = inrange_ny + 1;
    auto nz         = inrange_nz + 1;
    return dim3(nx, ny, nz);
};

}  // namespace

// malloc at function call
void spline3_configure(dim3 in_size3, size_t* quantcode_len, size_t* anchor_len)
{
    *quantcode_len = get_len_from_dim3(get_padded3(in_size3, get_nblock3(in_size3)));
    *anchor_len    = get_len_from_dim3(get_nanchor3(in_size3));
}

// accept device pointer, whose size is known
// internal data partition is fixed
template <typename DataIter, typename QuantIter, typename FP>
void compress_spline3(DataIter in, dim3 in_size3, DataIter anchor, QuantIter out, dim3 out_size, FP eb)
{
    using Data  = typename std::remove_pointer<DataIter*>::type;
    using Quant = typename std::remove_pointer<QuantIter*>::type;

    auto nblk3       = get_nblock3(in_size3);
    auto in_stride3  = get_stride3(in_size3);
    auto out_size3   = get_padded3(in_size3, nblk3);
    auto out_stride3 = get_stride3(out_size3);
    auto out_len     = get_len_from_dim3(out_size3);
    auto ac_size3    = get_nanchor3(in_size3);
    auto ac_stride3  = get_stride3(ac_size3);

    {
        /****************************************
         * anchor partition, e.g.
         *   localx = (235 + 7) / 8 (30)
         *   localy = (449 + 7) / 8 (57)
         *   localz = (449 + 7) / 8 (57)
         ****************************************/
        constexpr auto LINEAR_BLOCK_SIZE = 128;

        auto part_dim3 = dim3(ac_size3.x, 4, 4);
        auto dim_block = dim3(LINEAR_BLOCK_SIZE, 1, 1);
        auto dim_grid  = dim3(1, get_npart(ac_size3.y, 4), get_npart(ac_size3.z, 4));

        spline3d_anchors<Data, LINEAR_BLOCK_SIZE, true /*GATHER*/>
            <<<dim_grid, dim_block>>>(in, in_size3, in_stride3, anchor, ac_stride3);
    }
}

template <typename DataIter, typename QuantIter>
void decompress_spline3(dim3 in_size3)
{
    {
        /****************************************
         * anchor partition, e.g.
         *   localx = (235 + 7) / 8 (30)
         *   localy = (449 + 7) / 8 (57)
         *   localz = (449 + 7) / 8 (57)
         ****************************************/

        constexpr auto LINEAR_BLOCK_SIZE = 128;

        auto part_dim3 = dim3(ac_size3.x, 4, 4);
        auto dim_block = dim3(LINEAR_BLOCK_SIZE, 1, 1);
        auto dim_grid  = dim3(1, get_npart(ac_size3.y, 4), get_npart(ac_size3.z, 4));

        spline3d_anchors<Data, LINEAR_BLOCK_SIZE, false /*scatter*/>
            <<<dim_grid, dim_block>>>(in, in_size3, in_stride3, anchor, ac_stride3);
    }
}