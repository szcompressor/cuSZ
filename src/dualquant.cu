/**
 * @file cusz_dualquant.cu
 * @author Jiannan Tian
 * @brief Dual-Quantization method of cuSZ.
 * @version 0.2
 * @date 2021-01-16
 * (create) 19-09-23; (release) 2020-09-20; (rev1) 2021-01-16
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
#include <cstddef>

#include "dualquant.cuh"
#include "metadata.hh"
#include "type_aliasing.hh"

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

namespace kernel_v2 = cusz::predictor_quantizer::v2;
namespace kernel_v3 = cusz::predictor_quantizer::v3;

// v2 ////////////////////////////////////////////////////////////

template <typename Data, typename Quant>
__global__ void kernel_v2::c_lorenzo_1d1l(lorenzo_zip ctx, Data* d, Quant* q)
{
    auto id = bix * bdx + tix;

    if (id < ctx.d0) {
        // prequant (fp presence)
        d[id] = round(d[id] * ctx.ebx2_r);
        __syncthreads();  // necessary to ensure correctness

        // postquant
        Data pred        = tix == 0 ? 0 : d[id - 1];
        Data delta       = d[id] - pred;
        bool quantizable = fabs(delta) < ctx.radius;
        auto _code       = static_cast<Quant>(delta + ctx.radius);
        __syncthreads();                    // (!) somehow necessary to ensure correctness
        d[id] = (1 - quantizable) * d[id];  // output; reuse data for outlier
        q[id] = quantizable * _code;
    }
}

template <typename Data, typename Quant>
__global__ void kernel_v2::x_lorenzo_1d1l(lorenzo_unzip ctx, Data* xd, Data* outlier, Quant* q)
{
    static const auto Block = MetadataTrait<1>::Block;

    auto b0 = bdx * bix + tix;

    if (b0 >= ctx.n_blk0) return;

    auto _idx0 = b0 * Block;

    for (auto i0 = 0; i0 < Block; i0++) {
        auto id = _idx0 + i0;
        if (id >= ctx.d0) continue;
        Data pred = id < _idx0 + 1 ? 0 : xd[id - 1];
        xd[id]    = q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(ctx.radius);
    }
    for (auto i0 = 0; i0 < Block; i0++) {
        size_t id = _idx0 + i0;
        if (id >= ctx.d0) continue;
        xd[id] *= ctx.ebx2;
    }
    // end of body //
}

template <typename Data, typename Quant>
__global__ void kernel_v2::x_lorenzo_2d1l(lorenzo_unzip ctx, Data* xd, Data* outlier, Quant* q)
{
    static const auto Block = MetadataTrait<2>::Block;

    Data s[Block + 1][Block + 1];  // try not use shared memory first
    memset(s, 0, (Block + 1) * (Block + 1) * sizeof(Data));

    auto b1 = bdy * biy + tiy, b0 = bdx * bix + tix;

    if (b1 >= ctx.n_blk1 or b0 >= ctx.n_blk0) return;

    auto _idx1 = b1 * Block, _idx0 = b0 * Block;

    for (auto i1 = 0; i1 < Block; i1++) {
        for (auto i0 = 0; i0 < Block; i0++) {
            auto gi1 = _idx1 + i1, gi0 = _idx0 + i0;

            if (gi1 >= ctx.d1 or gi0 >= ctx.d0) continue;
            size_t id   = gi0 + gi1 * ctx.stride1;
            Data   pred = s[i1][i0 + 1] + s[i1 + 1][i0] - s[i1][i0];
            s[i1 + 1][i0 + 1] =
                q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(ctx.radius);
            xd[id] = s[i1 + 1][i0 + 1] * ctx.ebx2;
        }
    }
    // end of body //
}

template <typename Data, typename Quant>
__global__ void kernel_v2::x_lorenzo_3d1l(lorenzo_unzip ctx, Data* xd, Data* outlier, Quant* q)
{
    static const auto Block = MetadataTrait<3>::Block;

    Data s[Block + 1][Block + 1][Block + 1];
    memset(s, 0, (Block + 1) * (Block + 1) * (Block + 1) * sizeof(Data));

    auto b2 = bdz * biz + tiz, b1 = bdy * biy + tiy, b0 = bdx * bix + tix;

    if (b2 >= ctx.n_blk2 or b1 >= ctx.n_blk1 or b0 >= ctx.n_blk0) return;

    auto _idx2 = b2 * Block, _idx1 = b1 * Block, _idx0 = b0 * Block;

    for (auto i2 = 0; i2 < Block; i2++) {
        for (auto i1 = 0; i1 < Block; i1++) {
            for (auto i0 = 0; i0 < Block; i0++) {
                auto gi2 = _idx2 + i2, gi1 = _idx1 + i1, gi0 = _idx0 + i0;

                if (gi2 >= ctx.d2 or gi1 >= ctx.d1 or gi0 >= ctx.d0) continue;
                size_t id = gi0 + gi1 * ctx.stride1 + gi2 * ctx.stride2;

                Data pred = s[i2][i1][i0]                                                             // +, dist=3
                            - s[i2 + 1][i1][i0] - s[i2][i1 + 1][i0] - s[i2][i1][i0 + 1]               // -, dist=2
                            + s[i2 + 1][i1 + 1][i0] + s[i2 + 1][i1][i0 + 1] + s[i2][i1 + 1][i0 + 1];  // +, dist=1
                s[i2 + 1][i1 + 1][i0 + 1] =
                    q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(ctx.radius);
                xd[id] = s[i2 + 1][i1 + 1][i0 + 1] * ctx.ebx2;
            }
        }
    }
}

template __global__ void kernel_v2::c_lorenzo_1d1l<FP4, UI1>(lorenzo_zip, FP4*, UI1*);
template __global__ void kernel_v2::c_lorenzo_1d1l<FP4, UI2>(lorenzo_zip, FP4*, UI2*);

template __global__ void kernel_v2::x_lorenzo_1d1l<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel_v2::x_lorenzo_1d1l<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);
template __global__ void kernel_v2::x_lorenzo_2d1l<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel_v2::x_lorenzo_2d1l<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);
template __global__ void kernel_v2::x_lorenzo_3d1l<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel_v2::x_lorenzo_3d1l<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);

// v3 ////////////////////////////////////////////////////////////

template <typename Data, typename Quant>
__global__ void kernel_v3::c_lorenzo_2d1l(lorenzo_zip c, Data* d, Quant* q)
{
    static const auto Block   = MetadataTrait<2>::Block;
    Data(&s2df)[Block][Block] = *reinterpret_cast<Data(*)[Block][Block]>(&scratch);

    auto y = tiy, x = tix;
    auto gi1 = biy * bdy + y, gi0 = bix * bdx + x;

    if (gi0 < c.d0 and gi1 < c.d1) {
        size_t id = gi0 + gi1 * c.stride1;  // low to high dim, inner to outer

        // prequant (fp presence)
        s2df[y][x] = round(d[id] * c.ebx2_r);
        __syncthreads();  // necessary to ensure correctness

        Data delta = s2df[y][x] - ((x > 0 ? s2df[y][x - 1] : 0) +                // dist=1
                                   (y > 0 ? s2df[y - 1][x] : 0) -                // dist=1
                                   (x > 0 and y > 0 ? s2df[y - 1][x - 1] : 0));  // dist=2

        bool quantizable = fabs(delta) < c.radius;
        auto _code       = static_cast<Quant>(delta + c.radius);
        d[id]            = (1 - quantizable) * s2df[y][x];  // output; reuse data for outlier
        q[id]            = quantizable * _code;
    }
}

template <typename Data, typename Quant>
__global__ void kernel_v3::c_lorenzo_3d1l(lorenzo_zip ctx, Data* d, Quant* q)
{
    static const auto Block          = MetadataTrait<3>::Block;
    Data(&s3df)[Block][Block][Block] = *reinterpret_cast<Data(*)[Block][Block][Block]>(&scratch);

    auto z = tiz, y = tiy, x = tix;
    auto gi2 = biz * bdz + z, gi1 = biy * bdy + y, gi0 = bix * bdx + x;

    if (gi0 < ctx.d0 and gi1 < ctx.d1 and gi2 < ctx.d2) {
        size_t id = gi0 + gi1 * ctx.stride1 + gi2 * ctx.stride2;  // low to high in dim, inner to outer

        // prequant (fp presence)
        s3df[z][y][x] = round(d[id] * ctx.ebx2_r);
        __syncthreads();  // necessary to ensure correctness

        Data delta       = s3df[z][y][x] - ((z > 0 and y > 0 and x > 0 ? s3df[z - 1][y - 1][x - 1] : 0)  // dist=3
                                      - (y > 0 and x > 0 ? s3df[z][y - 1][x - 1] : 0)              // dist=2
                                      - (z > 0 and x > 0 ? s3df[z - 1][y][x - 1] : 0)              //
                                      - (z > 0 and y > 0 ? s3df[z - 1][y - 1][x] : 0)              //
                                      + (x > 0 ? s3df[z][y][x - 1] : 0)                            // dist=1
                                      + (y > 0 ? s3df[z][y - 1][x] : 0)                            //
                                      + (z > 0 ? s3df[z - 1][y][x] : 0));                          //
        bool quantizable = fabs(delta) < ctx.radius;
        auto _code       = static_cast<Quant>(delta + ctx.radius);
        d[id]            = (1 - quantizable) * s3df[z][y][x];  // output; reuse data for outlier
        q[id]            = quantizable * _code;
    }
}

template <typename Data, typename Quant>
__global__ void kernel_v3::x_lorenzo_2d1l(lorenzo_unzip ctx, Data* xd, Data* outlier, Quant* q)
{
    static const auto Block = MetadataTrait<2>::Block;

    Data s[Block][Block];  // try not use shared memory first
    memset(s, 0, Block * Block * sizeof(Data));

    auto b1 = bdy * biy + tiy, b0 = bdx * bix + tix;

    if (b1 >= ctx.n_blk1 or b0 >= ctx.n_blk0) return;

    auto _idx1 = b1 * Block, _idx0 = b0 * Block;

    for (auto i1 = 0; i1 < Block; i1++) {
        for (auto i0 = 0; i0 < Block; i0++) {
            auto gi1 = _idx1 + i1, gi0 = _idx0 + i0;

            if (gi1 >= ctx.d1 or gi0 >= ctx.d0) continue;
            size_t id   = gi0 + gi1 * ctx.stride1;
            Data   pred = (i1 > 0 ? s[i1 - 1][i0] : 0)  //
                        + (i0 > 0 ? s[i1][i0 - 1] : 0)  //
                        - (i1 > 0 and i0 > 0 ? s[i1 - 1][i0 - 1] : 0);
            s[i1][i0] = q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(ctx.radius);
            xd[id]    = s[i1][i0] * ctx.ebx2;
        }
    }
    // end of body //
}

template <typename Data, typename Quant>
__global__ void kernel_v3::x_lorenzo_3d1l(lorenzo_unzip ctx, Data* xd, Data* outlier, Quant* q)
{
    static const auto Block = MetadataTrait<3>::Block;

    Data s[Block][Block][Block];
    memset(s, 0, Block * Block * Block * sizeof(Data));

    auto b2 = bdz * biz + tiz, b1 = bdy * biy + tiy, b0 = bdx * bix + tix;

    if (b2 >= ctx.n_blk2 or b1 >= ctx.n_blk1 or b0 >= ctx.n_blk0) return;

    auto _idx2 = b2 * Block, _idx1 = b1 * Block, _idx0 = b0 * Block;

    for (auto i2 = 0; i2 < Block; i2++) {
        for (auto i1 = 0; i1 < Block; i1++) {
            for (auto i0 = 0; i0 < Block; i0++) {
                auto gi2 = _idx2 + i2, gi1 = _idx1 + i1, gi0 = _idx0 + i0;

                if (gi2 >= ctx.d2 or gi1 >= ctx.d1 or gi0 >= ctx.d0) continue;
                size_t id = gi0 + gi1 * ctx.stride1 + gi2 * ctx.stride2;

                Data pred = (i2 > 0 and i1 > 0 and i0 > 0 ? s[i2 - 1][i1 - 1][i0 - 1] : 0)  // dist=3
                            - (i1 > 0 and i0 > 0 ? s[i2][i1 - 1][i0 - 1] : 0)               // dist=2
                            - (i2 > 0 and i0 > 0 ? s[i2 - 1][i1][i0 - 1] : 0)               //
                            - (i2 > 0 and i1 > 0 ? s[i2 - 1][i1 - 1][i0] : 0)               //
                            + (i0 > 0 ? s[i2][i1][i0 - 1] : 0)                              // dist=1
                            + (i1 > 0 ? s[i2][i1 - 1][i0] : 0)                              //
                            + (i2 > 0 ? s[i2 - 1][i1][i0] : 0);                             //
                s[i2][i1][i0] =
                    q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(ctx.radius);
                xd[id] = s[i2][i1][i0] * ctx.ebx2;
            }
        }
    }
}

template __global__ void kernel_v3::c_lorenzo_2d1l<FP4, UI1>(lorenzo_zip, FP4*, UI1*);
template __global__ void kernel_v3::c_lorenzo_2d1l<FP4, UI2>(lorenzo_zip, FP4*, UI2*);
template __global__ void kernel_v3::c_lorenzo_3d1l<FP4, UI1>(lorenzo_zip, FP4*, UI1*);
template __global__ void kernel_v3::c_lorenzo_3d1l<FP4, UI2>(lorenzo_zip, FP4*, UI2*);

template __global__ void kernel_v3::x_lorenzo_2d1l<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel_v3::x_lorenzo_2d1l<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);
template __global__ void kernel_v3::x_lorenzo_3d1l<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel_v3::x_lorenzo_3d1l<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);