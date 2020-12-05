/**
 * @file cusz_dualquant.cu
 * @author Jiannan Tian
 * @brief Dual-Quantization method of cuSZ.
 * @version 0.1
 * @date 2020-09-20
 * Created on 19-09-23
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
#include <stdio.h>  // CUDA use
#include <cstddef>
#include <cstdint>

#include "cusz_dualquant.cuh"
#include "type_aliasing.hh"

const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
// const int DIM3   = 3;
const int nBLK0 = 4;
const int nBLK1 = 5;
const int nBLK2 = 6;
// const int nBLK3  = 7;
// const int nDIM   = 8;
// const int LEN    = 12;
// const int CAP    = 13;
const int RADIUS = 14;
// const size_t EB     = 0;
// const size_t EBr    = 1;
// const size_t EBx2   = 2;
const size_t EBx2_r = 3;

// extern __constant__ int    symb_dims[16];
// extern __constant__ double symb_ebs[4];

template <typename Data, typename Quant, int B>
__global__ void
cusz::predictor_quantizer::c_lorenzo_1d1l(Data* d, Quant* q, size_t const* dims, double const* precisions)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= dims[DIM0]) return;
    // prequant
    d[id] = round(d[id] * precisions[EBx2_r]);  // maintain fp representation
    __syncthreads();
    // postquant
    Data pred        = threadIdx.x == 0 ? 0 : d[id - 1];
    Data delta       = d[id] - pred;
    bool quantizable = fabs(delta) < dims[RADIUS];
    auto _code       = static_cast<Quant>(delta + dims[RADIUS]);
    __syncthreads();
    d[id] = (1 - quantizable) * d[id];  // data array as outlier
    q[id] = quantizable * _code;
}

template <typename Data, typename Quant, int B>
__global__ void
cusz::predictor_quantizer::c_lorenzo_2d1l(Data* d, Quant* q, size_t const* dims, double const* precisions)
{
    int y   = threadIdx.y;
    int x   = threadIdx.x;
    int gi1 = blockIdx.y * blockDim.y + y;
    int gi0 = blockIdx.x * blockDim.x + x;

    Data(&s2df)[B + 1][B + 1] = *reinterpret_cast<Data(*)[B + 1][B + 1]>(&scratch);
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

template <typename Data, typename Quant, int B>
__global__ void
cusz::predictor_quantizer::c_lorenzo_3d1l(Data* d, Quant* q, size_t const* dims, double const* precisions)
{
    int z = threadIdx.z;
    int y = threadIdx.y;
    int x = threadIdx.x;

    Data(&s3df)[B + 1][B + 1][B + 1] = *reinterpret_cast<Data(*)[B + 1][B + 1][B + 1]>(&scratch);

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

    int gi2 = blockIdx.z * blockDim.z + z;
    int gi1 = blockIdx.y * blockDim.y + y;
    int gi0 = blockIdx.x * blockDim.x + x;
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

template <typename Data, typename Quant, int B>
__global__ void
cusz::predictor_quantizer::x_lorenzo_1d1l(Data* xd, Data* outlier, Quant* q, size_t const* dims, double val_2eb)
{
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t b0 = blockDim.x * blockIdx.x + threadIdx.x;
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

template <typename Data, typename Quant, int B>
__global__ void
cusz::predictor_quantizer::x_lorenzo_2d1l(Data* xd, Data* outlier, Quant* q, size_t const* dims, double val_2eb)
{
    Data s[B + 1][B + 1];  // try not use shared memory first
    memset(s, 0, (B + 1) * (B + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t b1 = blockDim.y * blockIdx.y + threadIdx.y;
    size_t b0 = blockDim.x * blockIdx.x + threadIdx.x;

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

template <typename Data, typename Quant, int B>
__global__ void
cusz::predictor_quantizer::x_lorenzo_3d1l(Data* xd, Data* outlier, Quant* q, size_t const* dims, double val_2eb)
{
    Data s[B + 1][B + 1][B + 1];
    memset(s, 0, (B + 1) * (B + 1) * (B + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t b2 = blockDim.z * blockIdx.z + threadIdx.z;
    size_t b1 = blockDim.y * blockIdx.y + threadIdx.y;
    size_t b0 = blockDim.x * blockIdx.x + threadIdx.x;

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
// prototype 1D
// template __global__ void kernel::c_lorenzo_1d1l<FP4, UI1, 32>(FP4*, UI1*, size_t const*, FP8 const*);
// template __global__ void kernel::c_lorenzo_1d1l<FP4, UI2, 32>(FP4*, UI2*, size_t const*, FP8 const*);
// template __global__ void kernel::c_lorenzo_1d1l<FP4, UI1, 64>(FP4*, UI1*, size_t const*, FP8 const*);
// template __global__ void kernel::c_lorenzo_1d1l<FP4, UI2, 64>(FP4*, UI2*, size_t const*, FP8 const*);
// template __global__ void kernel::c_lorenzo_1d1l<FP4, UI1, 128>(FP4*, UI1*, size_t const*, FP8 const*);
// template __global__ void kernel::c_lorenzo_1d1l<FP4, UI2, 128>(FP4*, UI2*, size_t const*, FP8 const*);
template __global__ void kernel::c_lorenzo_1d1l<FP4, UI1, 256>(FP4*, UI1*, size_t const*, FP8 const*);
template __global__ void kernel::c_lorenzo_1d1l<FP4, UI2, 256>(FP4*, UI2*, size_t const*, FP8 const*);
template __global__ void kernel::c_lorenzo_1d1l<FP4, UI1, 512>(FP4*, UI1*, size_t const*, FP8 const*);
template __global__ void kernel::c_lorenzo_1d1l<FP4, UI2, 512>(FP4*, UI2*, size_t const*, FP8 const*);
template __global__ void kernel::c_lorenzo_1d1l<FP4, UI1, 1024>(FP4*, UI1*, size_t const*, FP8 const*);
template __global__ void kernel::c_lorenzo_1d1l<FP4, UI2, 1024>(FP4*, UI2*, size_t const*, FP8 const*);
// prototype 2D
template __global__ void kernel::c_lorenzo_2d1l<FP4, UI1, 16>(FP4*, UI1*, size_t const*, FP8 const*);
template __global__ void kernel::c_lorenzo_2d1l<FP4, UI2, 16>(FP4*, UI2*, size_t const*, FP8 const*);
// prototype 3D
template __global__ void kernel::c_lorenzo_3d1l<FP4, UI1, 8>(FP4*, UI1*, size_t const*, FP8 const*);
template __global__ void kernel::c_lorenzo_3d1l<FP4, UI2, 8>(FP4*, UI2*, size_t const*, FP8 const*);
// decompression
// prototype 1D
// template __global__ void kernel::x_lorenzo_1d1l<FP4, UI1, 32>(FP4*, FP4*, UI1*, size_t const*, FP8);
// template __global__ void kernel::x_lorenzo_1d1l<FP4, UI2, 32>(FP4*, FP4*, UI2*, size_t const*, FP8);
// template __global__ void kernel::x_lorenzo_1d1l<FP4, UI1, 64>(FP4*, FP4*, UI1*, size_t const*, FP8);
// template __global__ void kernel::x_lorenzo_1d1l<FP4, UI2, 64>(FP4*, FP4*, UI2*, size_t const*, FP8);
// template __global__ void kernel::x_lorenzo_1d1l<FP4, UI1, 128>(FP4*, FP4*, UI1*, size_t const*, FP8);
// template __global__ void kernel::x_lorenzo_1d1l<FP4, UI2, 128>(FP4*, FP4*, UI2*, size_t const*, FP8);
template __global__ void kernel::x_lorenzo_1d1l<FP4, UI1, 256>(FP4*, FP4*, UI1*, size_t const*, FP8);
template __global__ void kernel::x_lorenzo_1d1l<FP4, UI2, 256>(FP4*, FP4*, UI2*, size_t const*, FP8);
template __global__ void kernel::x_lorenzo_1d1l<FP4, UI1, 512>(FP4*, FP4*, UI1*, size_t const*, FP8);
template __global__ void kernel::x_lorenzo_1d1l<FP4, UI2, 512>(FP4*, FP4*, UI2*, size_t const*, FP8);
template __global__ void kernel::x_lorenzo_1d1l<FP4, UI1, 1024>(FP4*, FP4*, UI1*, size_t const*, FP8);
template __global__ void kernel::x_lorenzo_1d1l<FP4, UI2, 1024>(FP4*, FP4*, UI2*, size_t const*, FP8);
// prototype 2D
template __global__ void kernel::x_lorenzo_2d1l<FP4, UI1, 16>(FP4*, FP4*, UI1*, size_t const*, FP8);
template __global__ void kernel::x_lorenzo_2d1l<FP4, UI2, 16>(FP4*, FP4*, UI2*, size_t const*, FP8);
// prototype 3D
template __global__ void kernel::x_lorenzo_3d1l<FP4, UI1, 8>(FP4*, FP4*, UI1*, size_t const*, FP8);
template __global__ void kernel::x_lorenzo_3d1l<FP4, UI2, 8>(FP4*, FP4*, UI2*, size_t const*, FP8);
