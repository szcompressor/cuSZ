//
// Created by JianNan Tian on 9/23/19.
//

#ifndef CUSZ_DUALQUANT_CUH
#define CUSZ_DUALQUANT_CUH

#include <cuda_runtime.h>
#include <stdio.h>  // CUDA use

#include <cstddef>
#include "types.hh"

extern __shared__ float __s2df[][B_2d + 1];  // TODO double type
extern __shared__ float __s3df[][B_3d + 1][B_3d + 1];

namespace cuSZ {
// namespace PredictionDualQuantization {

namespace DryRun {
template <typename T>
__global__ void lorenzo_1d1l(T* data, size_t* dims_L16, double* ebs_L4) {
    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= dims_L16[DIM0]) return;
    data[id] = round(data[id] * ebs_L4[EBx2_r]) * ebs_L4[EBx2];  // prequantization
}

template <typename T>
__global__ void lorenzo_2d1l(T* data, size_t* dims_L16, double* ebs_L4) {
    auto   y   = threadIdx.y;
    auto   x   = threadIdx.x;
    auto   gi1 = blockIdx.y * blockDim.y + y;
    auto   gi0 = blockIdx.x * blockDim.x + x;
    size_t id  = gi0 + gi1 * dims_L16[DIM0];  // low to high dim, inner to outer
    if (gi0 >= dims_L16[DIM0] or gi1 >= dims_L16[DIM1]) return;
    data[id] = round(data[id] * ebs_L4[EBx2_r]) * ebs_L4[EBx2];  // prequantization
}

template <typename T>
__global__ void lorenzo_3d1l(T* data, size_t* dims_L16, double* ebs_L4) {
    auto   gi2 = blockIdx.z * blockDim.z + threadIdx.z;
    auto   gi1 = blockIdx.y * blockDim.y + threadIdx.y;
    auto   gi0 = blockIdx.x * blockDim.x + threadIdx.x;
    size_t id  = gi0 + gi1 * dims_L16[DIM0] + gi2 * dims_L16[DIM0] * dims_L16[DIM1];  // low to high in dim, inner to outer
    if (gi0 >= dims_L16[DIM0] or gi1 >= dims_L16[DIM1] or gi2 >= dims_L16[DIM2]) return;
    data[id] = round(data[id] * ebs_L4[EBx2_r]) * ebs_L4[EBx2];  // prequantization
}

}  // namespace DryRun

namespace PdQ {

template <typename T, typename Q, int B>
__global__ void c_lorenzo_1d1l(T* data, Q* code, size_t* dims_L16, double* ebs_L4) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= dims_L16[DIM0]) return;
    // prequantization
    data[id] = round(data[id] * ebs_L4[EBx2_r]);  // maintain fp representation
    __syncthreads();
    // postquantization
    T    pred        = threadIdx.x == 0 ? 0 : data[id - 1];
    T    posterror   = data[id] - pred;
    bool quantizable = fabs(posterror) < dims_L16[RADIUS];
    Q    _code       = static_cast<Q>(posterror + dims_L16[RADIUS]);
    __syncthreads();
    data[id] = (1 - quantizable) * data[id];  // data array as outlier
    code[id] = quantizable * _code;
}

template <typename T, typename Q, int B>
__global__ void c_lorenzo_2d1l(T* data, Q* code, size_t* dims_L16, double* ebs_L4) {
    int y = threadIdx.y;
    int x = threadIdx.x;

    if (x < B + 1 and y < B + 1) __s2df[y + 1][0] = 0, __s2df[0][x + 1] = 0;
    if (x == 0 and y == 0) __s2df[0][0] = 0;
    __syncthreads();

    int gi1 = blockIdx.y * blockDim.y + y;
    int gi0 = blockIdx.x * blockDim.x + x;
    if (gi0 >= dims_L16[DIM0] or gi1 >= dims_L16[DIM1]) return;
    size_t id = gi0 + gi1 * dims_L16[DIM0];  // low to high dim, inner to outer
    // prequantization
    __s2df[y + 1][x + 1] = round(data[id] * ebs_L4[EBx2_r]);  // fp representation
    __syncthreads();
    // postquantization
    T    pred        = __s2df[y + 1][x] + __s2df[y][x + 1] - __s2df[y][x];
    T    posterror   = __s2df[y + 1][x + 1] - pred;
    bool quantizable = fabs(posterror) < dims_L16[RADIUS];
    Q    _code       = static_cast<Q>(posterror + dims_L16[RADIUS]);
    __syncthreads();
    data[id] = (1 - quantizable) * __s2df[y + 1][x + 1];  // data array as outlier
    code[id] = quantizable * _code;
}

template <typename T, typename Q, int B>
__global__ void c_lorenzo_3d1l(T* data, Q* code, size_t* dims_L16, double* ebs_L4) {
    int z = threadIdx.z;
    int y = threadIdx.y;
    int x = threadIdx.x;

    if (x == 0) {
        __s3df[z + 1][y + 1][0] = 0;
        __s3df[0][z + 1][y + 1] = 0;
        __s3df[y + 1][0][z + 1] = 0;
    }
    if (x == 0 and y == 0) {
        __s3df[z + 1][0][0] = 0;
        __s3df[0][z + 1][0] = 0;
        __s3df[0][0][z + 1] = 0;
    }
    if (x == 0 and y == 0 and z == 0) __s3df[0][0][0] = 0;
    __syncthreads();

    int gi2 = blockIdx.z * blockDim.z + z;
    int gi1 = blockIdx.y * blockDim.y + y;
    int gi0 = blockIdx.x * blockDim.x + x;
    if (gi0 >= dims_L16[DIM0] or gi1 >= dims_L16[DIM1] or gi2 >= dims_L16[DIM2]) return;
    size_t id = gi0 + gi1 * dims_L16[DIM0] + gi2 * dims_L16[DIM0] * dims_L16[DIM1];  // low to high in dim, inner to outer
    // prequantization
    __s3df[z + 1][y + 1][x + 1] = round(data[id] * ebs_L4[EBx2_r]);  // fp representation
    __syncthreads();
    // postquantization
    T pred = __s3df[z][y][x]                                                                 // dist=3
             - __s3df[z + 1][y][x] - __s3df[z][y + 1][x] - __s3df[z][y][x + 1]               // dist=2
             + __s3df[z + 1][y + 1][x] + __s3df[z + 1][y][x + 1] + __s3df[z][y + 1][x + 1];  // dist=1
    T    posterror   = __s3df[z + 1][y + 1][x + 1] - pred;
    bool quantizable = fabs(posterror) < dims_L16[RADIUS];
    Q    _code       = static_cast<Q>(posterror + dims_L16[RADIUS]);
    __syncthreads();
    data[id] = (1 - quantizable) * __s3df[z + 1][y + 1][x + 1];  // data array as outlier
    code[id] = quantizable * _code;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//   ^                 decompression |
//   |compression                    v
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename Q, int B, int rounds = 2>
__global__ void x_lorenzo_1d1l(T* xdata, T* outlier, Q* bcode, size_t* dims_L16, double __2EB) {
    auto radius = static_cast<Q>(dims_L16[RADIUS]);

    size_t b0 = blockDim.x * blockIdx.x + threadIdx.x;
    if (b0 >= dims_L16[nBLK0]) return;
    size_t _idx0 = b0 * B;

    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims_L16[DIM0]) continue;
        T pred    = id < _idx0 + 1 ? 0 : xdata[id - 1];
        xdata[id] = bcode[id] == 0 ? outlier[id] : pred + static_cast<T>(bcode[id]) - static_cast<T>(radius);
    }
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims_L16[DIM0]) continue;
        xdata[id] *= __2EB;
    }
    // end of body //
}

template <typename T, typename Q, int B, int rounds = 2>
__global__ void x_lorenzo_2d1l(T* xdata, T* outlier, Q* bcode, size_t* dims_L16, double __2EB) {
    T __s[B + 1][B + 1];  // try not use shared memory first
    memset(__s, 0, (B + 1) * (B + 1) * sizeof(T));
    auto radius = static_cast<Q>(dims_L16[RADIUS]);

    size_t b1 = blockDim.y * blockIdx.y + threadIdx.y;
    size_t b0 = blockDim.x * blockIdx.x + threadIdx.x;

    if (b1 >= dims_L16[nBLK1] or b0 >= dims_L16[nBLK0]) return;

    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0]) continue;
            const size_t id     = gi0 + gi1 * dims_L16[DIM0];
            T            pred   = __s[i1][i0 + 1] + __s[i1 + 1][i0] - __s[i1][i0];
            __s[i1 + 1][i0 + 1] = bcode[id] == 0 ? outlier[id] : pred + static_cast<T>(bcode[id]) - static_cast<T>(radius);
            xdata[id]           = __s[i1 + 1][i0 + 1] * __2EB;
        }
    }
    // end of body //
}

template <typename T, typename Q, int B, int rounds = 2>
__global__ void x_lorenzo_3d1l(T*                  xdata,
                               T*                  outlier,
                               Q*                  bcode,
                               size_t const* const dims_L16,  //
                               double              __2EB) {
    T __s[B + 1][B + 1][B + 1];
    memset(__s, 0, (B + 1) * (B + 1) * (B + 1) * sizeof(T));
    auto radius = static_cast<Q>(dims_L16[RADIUS]);

    size_t b2 = blockDim.z * blockIdx.z + threadIdx.z;
    size_t b1 = blockDim.y * blockIdx.y + threadIdx.y;
    size_t b0 = blockDim.x * blockIdx.x + threadIdx.x;

    if (b2 >= dims_L16[nBLK2] or b1 >= dims_L16[nBLK1] or b0 >= dims_L16[nBLK0]) return;

    size_t _idx2 = b2 * B;
    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i2 = 0; i2 < B; i2++) {
        for (size_t i1 = 0; i1 < B; i1++) {
            for (size_t i0 = 0; i0 < B; i0++) {
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0;
                if (gi2 >= dims_L16[DIM2] or gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0]) continue;
                size_t id = gi0 + gi1 * dims_L16[DIM0] + gi2 * dims_L16[DIM1] * dims_L16[DIM0];

                T pred = __s[i2][i1][i0]                                                                 // +, dist=3
                         - __s[i2 + 1][i1][i0] - __s[i2][i1 + 1][i0] - __s[i2][i1][i0 + 1]               // -, dist=2
                         + __s[i2 + 1][i1 + 1][i0] + __s[i2 + 1][i1][i0 + 1] + __s[i2][i1 + 1][i0 + 1];  // +, dist=1
                __s[i2 + 1][i1 + 1][i0 + 1] = bcode[id] == 0 ? outlier[id] : pred + static_cast<T>(bcode[id]) - static_cast<T>(radius);
                xdata[id]                   = __s[i2 + 1][i1 + 1][i0 + 1] * __2EB;
            }
        }
    }
}

}  // namespace PdQ
}  // namespace cuSZ

#endif
