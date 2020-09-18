//
// Created by JianNan Tian on 9/23/19.
//

#include <cuda_runtime.h>
#include <stdio.h>  // CUDA use
#include <cstddef>
#include <cstdint>
#include "cusz_dualquant.cuh"

using uint8__t = uint8_t;

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

extern __constant__ int    symb_dims[16];
extern __constant__ double symb_ebs[4];

template <typename T, typename Q, int B>
__global__ void cusz::PdQ::c_lorenzo_1d1l(T* data, Q* code, size_t const* dims, double const* precisions)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= dims[DIM0]) return;
    // prequantization
    data[id] = round(data[id] * precisions[EBx2_r]);  // maintain fp representation
    __syncthreads();
    // postquantization
    T    pred        = threadIdx.x == 0 ? 0 : data[id - 1];
    T    posterror   = data[id] - pred;
    bool quantizable = fabs(posterror) < dims[RADIUS];
    Q    _code       = static_cast<Q>(posterror + dims[RADIUS]);
    __syncthreads();
    data[id] = (1 - quantizable) * data[id];  // data array as outlier
    code[id] = quantizable * _code;
}

// no new1
template <typename T, typename Q, int B>
__global__ void cusz::PdQ::c_lorenzo_1d1l_cmem(T* data, Q* code)
{
    auto   x    = threadIdx.x;
    size_t id   = blockIdx.x * blockDim.x + threadIdx.x;
    auto   s1df = reinterpret_cast<T*>(scratch);
    if (id >= symb_dims[DIM0]) return;
    // prequantization
    s1df[x] = round(data[id] * symb_ebs[EBx2_r]);  // maintain fp representation
    __syncthreads();
    // postquantization
    T    posterror   = s1df[x] - (x != 0 ? s1df[x - 1] : 0);
    bool quantizable = fabs(posterror) < symb_dims[RADIUS];
    __syncthreads();
    data[id] = (1 - quantizable) * s1df[x];  // data array as outlier
    code[id] = quantizable * static_cast<Q>(posterror + symb_dims[RADIUS]);
}

template <typename T, typename Q, int B>
__global__ void cusz::PdQ::c_lorenzo_2d1l(T* data, Q* code, size_t const* dims, double const* precisions)
{
    int y   = threadIdx.y;
    int x   = threadIdx.x;
    int gi1 = blockIdx.y * blockDim.y + y;
    int gi0 = blockIdx.x * blockDim.x + x;

    T(&s2df)[B + 1][B + 1] = *reinterpret_cast<T(*)[B + 1][B + 1]>(&scratch);
    if (x < B + 1 and y < B + 1) s2df[y + 1][0] = 0, s2df[0][x + 1] = 0;
    if (x == 0 and y == 0) s2df[0][0] = 0;
    __syncthreads();
    if (gi0 >= dims[DIM0] or gi1 >= dims[DIM1]) return;
    size_t id = gi0 + gi1 * dims[DIM0];  // low to high dim, inner to outer
    // prequantization
    s2df[y + 1][x + 1] = round(data[id] * precisions[EBx2_r]);  // fp representation
    __syncthreads();
    // postquantization
    T    pred        = s2df[y + 1][x] + s2df[y][x + 1] - s2df[y][x];
    T    posterror   = s2df[y + 1][x + 1] - pred;
    bool quantizable = fabs(posterror) < dims[RADIUS];
    Q    _code       = static_cast<Q>(posterror + dims[RADIUS]);
    __syncthreads();
    data[id] = (1 - quantizable) * s2df[y + 1][x + 1];  // data array as outlier
    code[id] = quantizable * _code;
}

/*
// use const memory
template <typename T, typename Q, int B>
__global__ void cusz::PdQ::c_lorenzo_2d1l_new(T* data, Q* code)
{
    auto y   = threadIdx.y;
    auto x   = threadIdx.x;
    auto gi1 = blockIdx.y * blockDim.y + y;
    auto gi0 = blockIdx.x * blockDim.x + x;

    T(&s2df)[B + 1][B + 1] = *reinterpret_cast<T(*)[B + 1][B + 1]>(&scratch);
    s2df[y + 1][0] = 0, s2df[0][x + 1] = 0, s2df[0][0] = 0;
    if (gi0 >= symb_dims[DIM0] or gi1 >= symb_dims[DIM1]) return;
    size_t id = gi0 + gi1 * symb_dims[DIM0];  // low to high dim, inner to outer
    // prequantization
    s2df[y + 1][x + 1] = round(data[id] * symb_ebs[EBx2_r]);  // fp representation
    __syncthreads();
    // postquantization
    T    posterror   = s2df[y + 1][x + 1] - s2df[y + 1][x] - s2df[y][x + 1] + s2df[y][x];
    bool quantizable = fabs(posterror) < symb_dims[RADIUS];
    code[id]         = quantizable * static_cast<Q>(posterror + symb_dims[RADIUS]);
    data[id]         = (1 - quantizable) * s2df[y + 1][x + 1];  // data array as outlier
}
 */

template <typename T, typename Q, int B>
__global__ void cusz::PdQ::c_lorenzo_2d1l_cmem(T* data, Q* code)
{
    auto y   = threadIdx.y;
    auto x   = threadIdx.x;
    auto gi1 = blockIdx.y * blockDim.y + y;
    auto gi0 = blockIdx.x * blockDim.x + x;

    T(&s2df)[B][B] = *reinterpret_cast<T(*)[B][B]>(&scratch);
    if (gi0 >= symb_dims[DIM0] or gi1 >= symb_dims[DIM1]) return;
    size_t id = gi0 + gi1 * symb_dims[DIM0];  // low to high dim, inner to outer
    // prequantization
    s2df[y][x] = round(data[id] * symb_ebs[EBx2_r]);  // fp representation
    __syncthreads();
    // postquantization
    T    posterror   = s2df[y][x] - (x == 0 ? 0 : s2df[y][x - 1]) - (y == 0 ? 0 : s2df[y - 1][x]) + (x > 0 and y > 0 ? s2df[y - 1][x - 1] : 0);
    bool quantizable = fabs(posterror) < symb_dims[RADIUS];
    code[id]         = quantizable * static_cast<Q>(posterror + symb_dims[RADIUS]);
    data[id]         = (1 - quantizable) * s2df[y][x];  // data array as outlier
}

template <typename T, typename Q, int B>
__global__ void cusz::PdQ::c_lorenzo_3d1l(T* data, Q* code, size_t const* dims, double const* precisions)
{
    int z = threadIdx.z;
    int y = threadIdx.y;
    int x = threadIdx.x;

    T(&s3df)[B + 1][B + 1][B + 1] = *reinterpret_cast<T(*)[B + 1][B + 1][B + 1]>(&scratch);

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
    // prequantization
    s3df[z + 1][y + 1][x + 1] = round(data[id] * precisions[EBx2_r]);  // fp representation
    __syncthreads();
    // postquantization
    T pred = s3df[z][y][x]                                                             // dist=3
             - s3df[z + 1][y][x] - s3df[z][y + 1][x] - s3df[z][y][x + 1]               // dist=2
             + s3df[z + 1][y + 1][x] + s3df[z + 1][y][x + 1] + s3df[z][y + 1][x + 1];  // dist=1
    T    posterror   = s3df[z + 1][y + 1][x + 1] - pred;
    bool quantizable = fabs(posterror) < dims[RADIUS];
    Q    _code       = static_cast<Q>(posterror + dims[RADIUS]);
    __syncthreads();
    data[id] = (1 - quantizable) * s3df[z + 1][y + 1][x + 1];  // data array as outlier
    code[id] = quantizable * _code;
}

/*
template <typename T, typename Q, int B>
__global__ void cusz::PdQ::c_lorenzo_3d1l_new(T* data, Q* code)
{
    auto z = threadIdx.z;
    auto y = threadIdx.y;
    auto x = threadIdx.x;

    T(&s3df)[B + 1][B + 1][B + 1] = *reinterpret_cast<T(*)[B + 1][B + 1][B + 1]>(&scratch);

    s3df[z + 1][y + 1][0] = 0;
    s3df[0][z + 1][y + 1] = 0;
    s3df[y + 1][0][z + 1] = 0;
    s3df[z + 1][0][0]     = 0;
    s3df[0][z + 1][0]     = 0;
    s3df[0][0][z + 1]     = 0;
    s3df[0][0][0]         = 0;

    auto gi2 = blockIdx.z * blockDim.z + z;
    auto gi1 = blockIdx.y * blockDim.y + y;
    auto gi0 = blockIdx.x * blockDim.x + x;

    if (gi0 >= symb_dims[DIM0] or gi1 >= symb_dims[DIM1] or gi2 >= symb_dims[DIM2]) return;
    size_t id = gi0 + gi1 * symb_dims[DIM0] + gi2 * symb_dims[DIM0] * symb_dims[DIM1];  // low to high in dim, inner to outer
    // prequantization
    s3df[z + 1][y + 1][x + 1] = round(data[id] * symb_ebs[EBx2_r]);  // fp representation
    __syncthreads();
    // postquantization
    T    posterror   = s3df[z + 1][y + 1][x + 1] - (                                                                              //
                                                  s3df[z][y][x]                                                              // dist=3
                                                  - s3df[z + 1][y][x] - s3df[z][y + 1][x] - s3df[z][y][x + 1]                // dist=2
                                                  + s3df[z + 1][y + 1][x] + s3df[z + 1][y][x + 1] + s3df[z][y + 1][x + 1]);  // dist=1
    bool quantizable = fabs(posterror) < symb_dims[RADIUS];
    code[id]         = quantizable * static_cast<Q>(posterror + symb_dims[RADIUS]);
    data[id]         = (1 - quantizable) * s3df[z + 1][y + 1][x + 1];  // data array as outlier
}
 */

template <typename T, typename Q, int B>
__global__ void cusz::PdQ::c_lorenzo_3d1l_cmem(T* data, Q* code)
{
    auto z   = threadIdx.z;
    auto y   = threadIdx.y;
    auto x   = threadIdx.x;
    auto gi2 = blockIdx.z * blockDim.z + z;
    auto gi1 = blockIdx.y * blockDim.y + y;
    auto gi0 = blockIdx.x * blockDim.x + x;

    T(&s3df)[B][B][B] = *reinterpret_cast<T(*)[B][B][B]>(&scratch);

    if (gi0 >= symb_dims[DIM0] or gi1 >= symb_dims[DIM1] or gi2 >= symb_dims[DIM2]) return;
    size_t id = gi0 + gi1 * symb_dims[DIM0] + gi2 * symb_dims[DIM0] * symb_dims[DIM1];  // low to high in dim, inner to outer
    // prequantization
    s3df[z][y][x] = round(data[id] * symb_ebs[EBx2_r]);  // fp representation
    __syncthreads();
    // postquantization
    T    posterror   = s3df[z][y][x] - (                                                                //
                                      (z > 0 and y > 0 and x > 0 ? s3df[z - 1][y - 1][x - 1] : 0)  // dist=3
                                      - (y > 0 and x > 0 ? s3df[z][y - 1][x - 1] : 0)              // dist=2
                                      - (z > 0 and x > 0 ? s3df[z - 1][y][x - 1] : 0)              //
                                      - (z > 0 and y > 0 ? s3df[z - 1][y - 1][x] : 0)              //
                                      + (x > 0 ? s3df[z][y][x - 1] : 0)                            // dist=1
                                      + (y > 0 ? s3df[z][y - 1][x] : 0)                            //
                                      + (z > 0 ? s3df[z - 1][y][x] : 0));                          //
    bool quantizable = fabs(posterror) < symb_dims[RADIUS];
    code[id]         = quantizable * static_cast<Q>(posterror + symb_dims[RADIUS]);
    data[id]         = (1 - quantizable) * s3df[z][y][x];  // data array as outlier
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//   ^                 decompression |
//   |compression                    v
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename Q, int B>
__global__ void cusz::PdQ::x_lorenzo_1d1l(T* xdata, T* outlier, Q* bcode, size_t const* dims, double val_2eb)
{
    auto radius = static_cast<Q>(dims[RADIUS]);

    size_t b0 = blockDim.x * blockIdx.x + threadIdx.x;
    if (b0 >= dims[nBLK0]) return;
    size_t _idx0 = b0 * B;

    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims[DIM0]) continue;
        T pred    = id < _idx0 + 1 ? 0 : xdata[id - 1];
        xdata[id] = bcode[id] == 0 ? outlier[id] : pred + static_cast<T>(bcode[id]) - static_cast<T>(radius);
    }
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims[DIM0]) continue;
        xdata[id] *= val_2eb;
    }
    // end of body //
}

template <typename T, typename Q, int B>
__global__ void cusz::PdQ::x_lorenzo_2d1l(T* xdata, T* outlier, Q* bcode, size_t const* dims, double val_2eb)
{
    T s[B + 1][B + 1];  // try not use shared memory first
    memset(s, 0, (B + 1) * (B + 1) * sizeof(T));
    auto radius = static_cast<Q>(dims[RADIUS]);

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
            T            pred = s[i1][i0 + 1] + s[i1 + 1][i0] - s[i1][i0];
            s[i1 + 1][i0 + 1] = bcode[id] == 0 ? outlier[id] : pred + static_cast<T>(bcode[id]) - static_cast<T>(radius);
            xdata[id]         = s[i1 + 1][i0 + 1] * val_2eb;
        }
    }
    // end of body //
}

template <typename T, typename Q, int B>
__global__ void cusz::PdQ::x_lorenzo_3d1l(T* xdata, T* outlier, Q* bcode, size_t const* dims, double val_2eb)
{
    T s[B + 1][B + 1][B + 1];
    memset(s, 0, (B + 1) * (B + 1) * (B + 1) * sizeof(T));
    auto radius = static_cast<Q>(dims[RADIUS]);

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

                T pred = s[i2][i1][i0]                                                             // +, dist=3
                         - s[i2 + 1][i1][i0] - s[i2][i1 + 1][i0] - s[i2][i1][i0 + 1]               // -, dist=2
                         + s[i2 + 1][i1 + 1][i0] + s[i2 + 1][i1][i0 + 1] + s[i2][i1 + 1][i0 + 1];  // +, dist=1
                s[i2 + 1][i1 + 1][i0 + 1] = bcode[id] == 0 ? outlier[id] : pred + static_cast<T>(bcode[id]) - static_cast<T>(radius);
                xdata[id]                 = s[i2 + 1][i1 + 1][i0 + 1] * val_2eb;
            }
        }
    }
}

// compression
// prototype 1D
template __global__ void cusz::PdQ::c_lorenzo_1d1l<float, uint8__t, 32>(float*, uint8__t*, size_t const*, double const*);
template __global__ void cusz::PdQ::c_lorenzo_1d1l<float, uint16_t, 32>(float*, uint16_t*, size_t const*, double const*);
template __global__ void cusz::PdQ::c_lorenzo_1d1l<float, uint32_t, 32>(float*, uint32_t*, size_t const*, double const*);
template __global__ void cusz::PdQ::c_lorenzo_1d1l<float, uint8__t, 64>(float*, uint8__t*, size_t const*, double const*);
template __global__ void cusz::PdQ::c_lorenzo_1d1l<float, uint16_t, 64>(float*, uint16_t*, size_t const*, double const*);
template __global__ void cusz::PdQ::c_lorenzo_1d1l<float, uint32_t, 64>(float*, uint32_t*, size_t const*, double const*);
// prototype 2D
template __global__ void cusz::PdQ::c_lorenzo_2d1l<float, uint8__t, 16>(float*, uint8__t*, size_t const*, double const*);
template __global__ void cusz::PdQ::c_lorenzo_2d1l<float, uint16_t, 16>(float*, uint16_t*, size_t const*, double const*);
template __global__ void cusz::PdQ::c_lorenzo_2d1l<float, uint32_t, 16>(float*, uint32_t*, size_t const*, double const*);
// prototype 3D
template __global__ void cusz::PdQ::c_lorenzo_3d1l<float, uint8__t, 8>(float*, uint8__t*, size_t const*, double const*);
template __global__ void cusz::PdQ::c_lorenzo_3d1l<float, uint16_t, 8>(float*, uint16_t*, size_t const*, double const*);
template __global__ void cusz::PdQ::c_lorenzo_3d1l<float, uint32_t, 8>(float*, uint32_t*, size_t const*, double const*);
// decompression
// prototype 1D
template __global__ void cusz::PdQ::x_lorenzo_1d1l<float, uint8__t, 32>(float*, float*, uint8__t*, size_t const*, double);
template __global__ void cusz::PdQ::x_lorenzo_1d1l<float, uint16_t, 32>(float*, float*, uint16_t*, size_t const*, double);
template __global__ void cusz::PdQ::x_lorenzo_1d1l<float, uint32_t, 32>(float*, float*, uint32_t*, size_t const*, double);
template __global__ void cusz::PdQ::x_lorenzo_1d1l<float, uint8__t, 64>(float*, float*, uint8__t*, size_t const*, double);
template __global__ void cusz::PdQ::x_lorenzo_1d1l<float, uint16_t, 64>(float*, float*, uint16_t*, size_t const*, double);
template __global__ void cusz::PdQ::x_lorenzo_1d1l<float, uint32_t, 64>(float*, float*, uint32_t*, size_t const*, double);
// prototype 2D
template __global__ void cusz::PdQ::x_lorenzo_2d1l<float, uint8__t, 16>(float*, float*, uint8__t*, size_t const*, double);
template __global__ void cusz::PdQ::x_lorenzo_2d1l<float, uint16_t, 16>(float*, float*, uint16_t*, size_t const*, double);
template __global__ void cusz::PdQ::x_lorenzo_2d1l<float, uint32_t, 16>(float*, float*, uint32_t*, size_t const*, double);
// prototype 3D
template __global__ void cusz::PdQ::x_lorenzo_3d1l<float, uint8__t, 8>(float*, float*, uint8__t*, size_t const*, double);
template __global__ void cusz::PdQ::x_lorenzo_3d1l<float, uint16_t, 8>(float*, float*, uint16_t*, size_t const*, double);
template __global__ void cusz::PdQ::x_lorenzo_3d1l<float, uint32_t, 8>(float*, float*, uint32_t*, size_t const*, double);

// c using const mem
template __global__ void cusz::PdQ::c_lorenzo_1d1l_cmem<float, uint8__t, 32>(float*, uint8__t*);
template __global__ void cusz::PdQ::c_lorenzo_1d1l_cmem<float, uint16_t, 32>(float*, uint16_t*);
template __global__ void cusz::PdQ::c_lorenzo_1d1l_cmem<float, uint32_t, 32>(float*, uint32_t*);
template __global__ void cusz::PdQ::c_lorenzo_1d1l_cmem<float, uint8__t, 64>(float*, uint8__t*);
template __global__ void cusz::PdQ::c_lorenzo_1d1l_cmem<float, uint16_t, 64>(float*, uint16_t*);
template __global__ void cusz::PdQ::c_lorenzo_1d1l_cmem<float, uint32_t, 64>(float*, uint32_t*);

template __global__ void cusz::PdQ::c_lorenzo_2d1l_cmem<float, uint8__t, 16>(float*, uint8__t*);
template __global__ void cusz::PdQ::c_lorenzo_2d1l_cmem<float, uint16_t, 16>(float*, uint16_t*);
template __global__ void cusz::PdQ::c_lorenzo_2d1l_cmem<float, uint32_t, 16>(float*, uint32_t*);

template __global__ void cusz::PdQ::c_lorenzo_3d1l_cmem<float, uint8__t, 8>(float*, uint8__t*);
template __global__ void cusz::PdQ::c_lorenzo_3d1l_cmem<float, uint16_t, 8>(float*, uint16_t*);
template __global__ void cusz::PdQ::c_lorenzo_3d1l_cmem<float, uint32_t, 8>(float*, uint32_t*);
