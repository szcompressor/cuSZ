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
#include <stdio.h>  // CUDA use
#include <cstddef>
#include <cstdint>

#include "dualquant.cuh"
#include "metadata.hh"
#include "type_aliasing.hh"

const int    DIM0   = 0;
const int    DIM1   = 1;
const int    DIM2   = 2;
const int    nBLK0  = 4;
const int    nBLK1  = 5;
const int    nBLK2  = 6;
const int    RADIUS = 14;
const size_t EBx2_r = 3;

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

template <typename Data, typename Quant>
__global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l(Data* d, Quant* q, size_t const* dims, double const* precisions)
{
    auto id = bix * bdx + tix;
    if (id >= dims[DIM0]) return;
    // prequant
    d[id] = round(d[id] * precisions[EBx2_r]);  // maintain fp representation
    __syncthreads();
    // postquant
    Data pred        = tix == 0 ? 0 : d[id - 1];
    Data delta       = d[id] - pred;
    bool quantizable = fabs(delta) < dims[RADIUS];
    auto _code       = static_cast<Quant>(delta + dims[RADIUS]);
    __syncthreads();
    d[id] = (1 - quantizable) * d[id];  // data array as outlier
    q[id] = quantizable * _code;
}

template <typename Data, typename Quant>
__global__ void
cusz::predictor_quantizer::c_lorenzo_2d1l(Data* d, Quant* q, size_t const* dims, double const* precisions)
{
    static const int B        = MetadataTrait<2>::Block;
    Data(&s2df)[B + 1][B + 1] = *reinterpret_cast<Data(*)[B + 1][B + 1]>(&scratch);

    auto y = tiy, x = tix;
    auto gi1 = biy * bdy + y, gi0 = bix * bdx + x;

    if (x < B + 1 and y < B + 1) s2df[y + 1][0] = 0, s2df[0][x + 1] = 0;
    if (x == 0 and y == 0) s2df[0][0] = 0;
    __syncthreads();
    if (gi0 >= dims[DIM0] or gi1 >= dims[DIM1]) return;
    size_t id = gi0 + gi1 * dims[DIM0];  // low to high dim, inner to outer
    // prequant
    s2df[y + 1][x + 1] = round(d[id] * precisions[EBx2_r]);  // fp representation
    __syncthreads();
    // postquant
    Data pred        = s2df[y + 1][x] + s2df[y][x + 1] - s2df[y][x];
    Data delta       = s2df[y + 1][x + 1] - pred;
    bool quantizable = fabs(delta) < dims[RADIUS];
    auto _code       = static_cast<Quant>(delta + dims[RADIUS]);
    __syncthreads();
    d[id] = (1 - quantizable) * s2df[y + 1][x + 1];  // data array as outlier
    q[id] = quantizable * _code;
}

template <typename Data, typename Quant>
__global__ void
cusz::predictor_quantizer::c_lorenzo_3d1l(Data* d, Quant* q, size_t const* dims, double const* precisions)
{
    static const int B = MetadataTrait<3>::Block;

    Data(&s3df)[B + 1][B + 1][B + 1] = *reinterpret_cast<Data(*)[B + 1][B + 1][B + 1]>(&scratch);

    auto z = tiz, y = tiy, x = tix;

    if (x == 0) {
        s3df[z + 1][y + 1][0] = 0;
        s3df[0][z + 1][y + 1] = 0;
        s3df[y + 1][0][z + 1] = 0;
    }
    if (x == 0 and y == 0) {
        s3df[z + 1][0][0] = 0;
        s3df[0][z + 1][0] = 0;
        s3df[0][0][z + 1] = 0;
    }
    if (x == 0 and y == 0 and z == 0) s3df[0][0][0] = 0;
    __syncthreads();

    auto gi2 = biz * bdz + z, gi1 = biy * bdy + y, gi0 = bix * bdx + x;

    if (gi0 >= dims[DIM0] or gi1 >= dims[DIM1] or gi2 >= dims[DIM2]) return;
    size_t id = gi0 + gi1 * dims[DIM0] + gi2 * dims[DIM0] * dims[DIM1];  // low to high in dim, inner to outer
    // prequant
    s3df[z + 1][y + 1][x + 1] = round(d[id] * precisions[EBx2_r]);  // fp representation
    __syncthreads();
    // postquant
    Data pred = s3df[z][y][x]                                                             // dist=3
                - s3df[z + 1][y][x] - s3df[z][y + 1][x] - s3df[z][y][x + 1]               // dist=2
                + s3df[z + 1][y + 1][x] + s3df[z + 1][y][x + 1] + s3df[z][y + 1][x + 1];  // dist=1
    Data delta       = s3df[z + 1][y + 1][x + 1] - pred;
    bool quantizable = fabs(delta) < dims[RADIUS];
    auto _code       = static_cast<Quant>(delta + dims[RADIUS]);
    __syncthreads();
    d[id] = (1 - quantizable) * s3df[z + 1][y + 1][x + 1];  // data array as outlier
    q[id] = quantizable * _code;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//   ^                 decompression |
//   |compression                    v
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Data, typename Quant>
__global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l(Data* xd, Data* outlier, Quant* q, size_t const* dims, double val_2eb)
{
    static const int B = MetadataTrait<1>::Block;

    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t b0 = bdx * bix + tix;
    if (b0 >= dims[nBLK0]) return;
    size_t _idx0 = b0 * B;

    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims[DIM0]) continue;
        Data pred = id < _idx0 + 1 ? 0 : xd[id - 1];
        xd[id]    = q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(radius);
    }
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims[DIM0]) continue;
        xd[id] *= val_2eb;
    }
    // end of body //
}

template <typename Data, typename Quant>
__global__ void
cusz::predictor_quantizer::x_lorenzo_2d1l(Data* xd, Data* outlier, Quant* q, size_t const* dims, double val_2eb)
{
    static const int B = MetadataTrait<2>::Block;

    Data s[B + 1][B + 1];  // try not use shared memory first
    memset(s, 0, (B + 1) * (B + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t b1 = bdy * biy + tiy;
    size_t b0 = bdx * bix + tix;

    if (b1 >= dims[nBLK1] or b0 >= dims[nBLK0]) return;

    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims[DIM1] or gi0 >= dims[DIM0]) continue;
            const size_t id   = gi0 + gi1 * dims[DIM0];
            Data         pred = s[i1][i0 + 1] + s[i1 + 1][i0] - s[i1][i0];
            s[i1 + 1][i0 + 1] = q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(radius);
            xd[id]            = s[i1 + 1][i0 + 1] * val_2eb;
        }
    }
    // end of body //
}

template <typename Data, typename Quant>
__global__ void
cusz::predictor_quantizer::x_lorenzo_3d1l(Data* xd, Data* outlier, Quant* q, size_t const* dims, double val_2eb)
{
    static const int B = MetadataTrait<3>::Block;

    Data s[B + 1][B + 1][B + 1];
    memset(s, 0, (B + 1) * (B + 1) * (B + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t b2 = bdz * biz + tiz;
    size_t b1 = bdy * biy + tiy;
    size_t b0 = bdx * bix + tix;

    if (b2 >= dims[nBLK2] or b1 >= dims[nBLK1] or b0 >= dims[nBLK0]) return;

    size_t _idx2 = b2 * B;
    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i2 = 0; i2 < B; i2++) {
        for (size_t i1 = 0; i1 < B; i1++) {
            for (size_t i0 = 0; i0 < B; i0++) {
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0;
                if (gi2 >= dims[DIM2] or gi1 >= dims[DIM1] or gi0 >= dims[DIM0]) continue;
                size_t id = gi0 + gi1 * dims[DIM0] + gi2 * dims[DIM1] * dims[DIM0];

                Data pred = s[i2][i1][i0]                                                             // +, dist=3
                            - s[i2 + 1][i1][i0] - s[i2][i1 + 1][i0] - s[i2][i1][i0 + 1]               // -, dist=2
                            + s[i2 + 1][i1 + 1][i0] + s[i2 + 1][i1][i0 + 1] + s[i2][i1 + 1][i0 + 1];  // +, dist=1
                s[i2 + 1][i1 + 1][i0 + 1] =
                    q[id] == 0 ? outlier[id] : pred + static_cast<Data>(q[id]) - static_cast<Data>(radius);
                xd[id] = s[i2 + 1][i1 + 1][i0 + 1] * val_2eb;
            }
        }
    }
}

namespace kernel = cusz::predictor_quantizer;

// compression
template __global__ void kernel::c_lorenzo_1d1l<FP4, UI1>(FP4*, UI1*, size_t const*, FP8 const*);
template __global__ void kernel::c_lorenzo_1d1l<FP4, UI2>(FP4*, UI2*, size_t const*, FP8 const*);
template __global__ void kernel::c_lorenzo_2d1l<FP4, UI1>(FP4*, UI1*, size_t const*, FP8 const*);
template __global__ void kernel::c_lorenzo_2d1l<FP4, UI2>(FP4*, UI2*, size_t const*, FP8 const*);
template __global__ void kernel::c_lorenzo_3d1l<FP4, UI1>(FP4*, UI1*, size_t const*, FP8 const*);
template __global__ void kernel::c_lorenzo_3d1l<FP4, UI2>(FP4*, UI2*, size_t const*, FP8 const*);
// decompression
template __global__ void kernel::x_lorenzo_1d1l<FP4, UI1>(FP4*, FP4*, UI1*, size_t const*, FP8);
template __global__ void kernel::x_lorenzo_1d1l<FP4, UI2>(FP4*, FP4*, UI2*, size_t const*, FP8);
template __global__ void kernel::x_lorenzo_2d1l<FP4, UI1>(FP4*, FP4*, UI1*, size_t const*, FP8);
template __global__ void kernel::x_lorenzo_2d1l<FP4, UI2>(FP4*, FP4*, UI2*, size_t const*, FP8);
template __global__ void kernel::x_lorenzo_3d1l<FP4, UI1>(FP4*, FP4*, UI1*, size_t const*, FP8);
template __global__ void kernel::x_lorenzo_3d1l<FP4, UI2>(FP4*, FP4*, UI2*, size_t const*, FP8);

// v2 ////////////////////////////////////////////////////////////

template <typename Data, typename Quant>
__global__ void cusz::predictor_quantizer::c_lorenzo_1d1l_v2(lorenzo_zip ctx, Data* d, Quant* q)
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
__global__ void cusz::predictor_quantizer::c_lorenzo_2d1l_v2(lorenzo_zip c, Data* d, Quant* q)
{
    static const int Block            = MetadataTrait<2>::Block;
    Data(&s2df)[Block + 1][Block + 1] = *reinterpret_cast<Data(*)[Block + 1][Block + 1]>(&scratch);

    auto y = tiy, x = tix;
    auto gi1 = biy * bdy + y, gi0 = bix * bdx + x;

    // reset cache
    if (x < Block + 1 and y < Block + 1) s2df[y + 1][0] = 0, s2df[0][x + 1] = 0;
    if (x == 0 and y == 0) s2df[0][0] = 0;
    __syncthreads();

    if (gi0 < c.d0 and gi1 < c.d1) {
        size_t id = gi0 + gi1 * c.stride1;  // low to high dim, inner to outer

        // prequant (fp presence)
        s2df[y + 1][x + 1] = round(d[id] * c.ebx2_r);
        __syncthreads();  // necessary to ensure correctness

        // postquant
        Data pred        = s2df[y + 1][x] + s2df[y][x + 1] - s2df[y][x];
        Data delta       = s2df[y + 1][x + 1] - pred;
        bool quantizable = fabs(delta) < c.radius;
        auto _code       = static_cast<Quant>(delta + c.radius);
        d[id]            = (1 - quantizable) * s2df[y + 1][x + 1];  // output; reuse data for outlier
        q[id]            = quantizable * _code;
    }
}

template <typename Data, typename Quant>
__global__ void cusz::predictor_quantizer::c_lorenzo_3d1l_v2(lorenzo_zip ctx, Data* d, Quant* q)
{
    static const int Block = MetadataTrait<3>::Block;

    Data(&s3df)[Block + 1][Block + 1][Block + 1] =
        *reinterpret_cast<Data(*)[Block + 1][Block + 1][Block + 1]>(&scratch);

    auto z = tiz, y = tiy, x = tix;

    // reset cache
    if (x == 0) {
        s3df[z + 1][y + 1][0] = 0;
        s3df[0][z + 1][y + 1] = 0;
        s3df[y + 1][0][z + 1] = 0;
    }
    if (x == 0 and y == 0) {
        s3df[z + 1][0][0] = 0;
        s3df[0][z + 1][0] = 0;
        s3df[0][0][z + 1] = 0;
    }
    if (x == 0 and y == 0 and z == 0) s3df[0][0][0] = 0;
    __syncthreads();

    auto gi2 = biz * bdz + z, gi1 = biy * bdy + y, gi0 = bix * bdx + x;

    if (gi0 < ctx.d0 and gi1 < ctx.d1 and gi2 < ctx.d2) {
        size_t id = gi0 + gi1 * ctx.stride1 + gi2 * ctx.stride2;  // low to high in dim, inner to outer

        // prequant (fp presence)
        s3df[z + 1][y + 1][x + 1] = round(d[id] * ctx.ebx2_r);
        __syncthreads();  // necessary to ensure correctness

        // postquant
        Data pred = s3df[z][y][x]                                                             // dist=3
                    - s3df[z + 1][y][x] - s3df[z][y + 1][x] - s3df[z][y][x + 1]               // dist=2
                    + s3df[z + 1][y + 1][x] + s3df[z + 1][y][x + 1] + s3df[z][y + 1][x + 1];  // dist=1
        Data delta       = s3df[z + 1][y + 1][x + 1] - pred;
        bool quantizable = fabs(delta) < ctx.radius;
        auto _code       = static_cast<Quant>(delta + ctx.radius);
        d[id]            = (1 - quantizable) * s3df[z + 1][y + 1][x + 1];  // output; reuse data for outlier
        q[id]            = quantizable * _code;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//   ^                 decompression |
//   |compression                    v
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Data, typename Quant>
__global__ void cusz::predictor_quantizer::x_lorenzo_1d1l_v2(lorenzo_unzip ctx, Data* xd, Data* outlier, Quant* q)
{
    static const int Block = MetadataTrait<1>::Block;

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
__global__ void cusz::predictor_quantizer::x_lorenzo_2d1l_v2(lorenzo_unzip ctx, Data* xd, Data* outlier, Quant* q)
{
    static const int Block = MetadataTrait<2>::Block;

    Data s[Block + 1][Block + 1];  // try not use shared memory first
    memset(s, 0, (Block + 1) * (Block + 1) * sizeof(Data));

    auto b1 = bdy * biy + tiy;
    auto b0 = bdx * bix + tix;

    if (b1 >= ctx.n_blk1 or b0 >= ctx.n_blk0) return;

    auto _idx1 = b1 * Block;
    auto _idx0 = b0 * Block;

    for (auto i1 = 0; i1 < Block; i1++) {
        for (auto i0 = 0; i0 < Block; i0++) {
            auto gi1 = _idx1 + i1;
            auto gi0 = _idx0 + i0;
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
__global__ void cusz::predictor_quantizer::x_lorenzo_3d1l_v2(lorenzo_unzip ctx, Data* xd, Data* outlier, Quant* q)
{
    static const int Block = MetadataTrait<3>::Block;

    Data s[Block + 1][Block + 1][Block + 1];
    memset(s, 0, (Block + 1) * (Block + 1) * (Block + 1) * sizeof(Data));

    auto b2 = bdz * biz + tiz;
    auto b1 = bdy * biy + tiy;
    auto b0 = bdx * bix + tix;

    if (b2 >= ctx.n_blk2 or b1 >= ctx.n_blk1 or b0 >= ctx.n_blk0) return;

    auto _idx2 = b2 * Block;
    auto _idx1 = b1 * Block;
    auto _idx0 = b0 * Block;

    for (auto i2 = 0; i2 < Block; i2++) {
        for (auto i1 = 0; i1 < Block; i1++) {
            for (auto i0 = 0; i0 < Block; i0++) {
                auto gi2 = _idx2 + i2;
                auto gi1 = _idx1 + i1;
                auto gi0 = _idx0 + i0;
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

template __global__ void kernel::c_lorenzo_1d1l_v2<FP4, UI1>(lorenzo_zip, FP4*, UI1*);
template __global__ void kernel::c_lorenzo_1d1l_v2<FP4, UI2>(lorenzo_zip, FP4*, UI2*);
template __global__ void kernel::c_lorenzo_2d1l_v2<FP4, UI1>(lorenzo_zip, FP4*, UI1*);
template __global__ void kernel::c_lorenzo_2d1l_v2<FP4, UI2>(lorenzo_zip, FP4*, UI2*);
template __global__ void kernel::c_lorenzo_3d1l_v2<FP4, UI1>(lorenzo_zip, FP4*, UI1*);
template __global__ void kernel::c_lorenzo_3d1l_v2<FP4, UI2>(lorenzo_zip, FP4*, UI2*);

template __global__ void kernel::x_lorenzo_1d1l_v2<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel::x_lorenzo_1d1l_v2<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);
template __global__ void kernel::x_lorenzo_2d1l_v2<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel::x_lorenzo_2d1l_v2<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);
template __global__ void kernel::x_lorenzo_3d1l_v2<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel::x_lorenzo_3d1l_v2<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);

// v3 ////////////////////////////////////////////////////////////

template <typename Data, typename Quant>
__global__ void cusz::predictor_quantizer::c_lorenzo_2d1l_v3(lorenzo_zip c, Data* d, Quant* q)
{
    static const int Block    = MetadataTrait<2>::Block;
    Data(&s2df)[Block][Block] = *reinterpret_cast<Data(*)[Block][Block]>(&scratch);

    auto y = tiy, x = tix;
    auto gi1 = biy * bdy + y, gi0 = bix * bdx + x;

    if (gi0 < c.d0 and gi1 < c.d1) {
        size_t id = gi0 + gi1 * c.stride1;  // low to high dim, inner to outer

        // prequant (fp presence)
        s2df[y][x] = round(d[id] * c.ebx2_r);
        __syncthreads();  // necessary to ensure correctness

        Data delta = s2df[y][x] - ((x == 0 ? 0 : s2df[y][x - 1]) +               // dist=1
                                   (y == 0 ? 0 : s2df[y - 1][x]) -               // dist=1
                                   (x > 0 and y > 0 ? s2df[y - 1][x - 1] : 0));  // dist=2

        bool quantizable = fabs(delta) < c.radius;
        auto _code       = static_cast<Quant>(delta + c.radius);
        d[id]            = (1 - quantizable) * s2df[y][x];  // output; reuse data for outlier
        q[id]            = quantizable * _code;
    }
}

template <typename Data, typename Quant>
__global__ void cusz::predictor_quantizer::c_lorenzo_3d1l_v3(lorenzo_zip ctx, Data* d, Quant* q)
{
    static const int Block = MetadataTrait<3>::Block;

    Data(&s3df)[Block][Block][Block] = *reinterpret_cast<Data(*)[Block][Block][Block]>(&scratch);

    auto z = tiz, y = tiy, x = tix;
    auto gi2 = biz * bdz + z, gi1 = biy * bdy + y, gi0 = bix * bdx + x;

    if (gi0 < ctx.d0 and gi1 < ctx.d1 and gi2 < ctx.d2) {
        size_t id = gi0 + gi1 * ctx.stride1 + gi2 * ctx.stride2;  // low to high in dim, inner to outer

        // prequant (fp presence)
        s3df[z][y][x] = round(d[id] * ctx.ebx2_r);
        __syncthreads();  // necessary to ensure correctness

        Data delta = s3df[z][y][x] - ((z > 0 and y > 0 and x > 0 ? s3df[z - 1][y - 1][x - 1] : 0)  // dist=3
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

template __global__ void kernel::c_lorenzo_2d1l_v3<FP4, UI1>(lorenzo_zip, FP4*, UI1*);
template __global__ void kernel::c_lorenzo_2d1l_v3<FP4, UI2>(lorenzo_zip, FP4*, UI2*);
template __global__ void kernel::c_lorenzo_3d1l_v3<FP4, UI1>(lorenzo_zip, FP4*, UI1*);
template __global__ void kernel::c_lorenzo_3d1l_v3<FP4, UI2>(lorenzo_zip, FP4*, UI2*);
