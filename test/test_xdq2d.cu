#if CUDART_VERSION >= 11000
#pragma message(__FILE__ ": (CUDA 11 onward), cub from system path")
#include <cub/cub.cuh>
#else
#pragma message(__FILE__ ": (CUDA 10 or earlier), cub from git submodule")
#include "../external/cub/cub/cub.cuh"
#endif

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>
using std::cout;
using std::endl;
#include "../src/dualquant.cuh"
#include "../src/metadata.hh"
#include "../src/type_aliasing.hh"

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

template <typename T>
struct data_t {
    typedef T Type;
    T*        h;
    T*        d;
    size_t    len;
    size_t    bytes;
};

using Data  = float;
using Quant = uint16_t;


int main()
{
    struct data_t<float> o;
    //    o.len   = 256 * 2;
    o.len   = 256;
    o.bytes = o.len * sizeof(decltype(o)::Type);

    struct data_t<uint16_t> q;
    //    q.len   = 256 * 2;
    q.len   = 256;
    q.bytes = q.len * sizeof(decltype(q)::Type);

    cudaMallocHost(&o.h, o.bytes);
    cudaMalloc(&o.d, o.bytes);

    cudaMallocHost(&q.h, q.bytes);
    cudaMalloc(&q.d, q.bytes);

    for (auto i = 0; i < o.len; i++) { q.h[i] = 1, o.h[i] = 0; }

    cudaMemcpy(o.d, o.h, o.bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(q.d, q.h, q.bytes, cudaMemcpyHostToDevice);

    LorenzoUnzipContext ctx;
    ctx.d0 = 16;
    //    ctx.d1      = 16 * 2;
    ctx.d1      = 16;
    ctx.stride1 = 16;
    ctx.radius  = 0;
    ctx.ebx2    = 1;

    //    if (cub)
//    explain_why_cub_nd_not_working  //
//        <<<dim3(1, 1), dim3(2, 16)>>>(ctx, o.d, o.d, q.d);
    //        else
    cusz::predictor_quantizer::x_lorenzo_2d1l_16x16_v1  //
        <decltype(o)::Type, decltype(q)::Type>          //
        <<<dim3(1, 1), dim3(16, 2)>>>(ctx, o.d, o.d, q.d);

    cudaDeviceSynchronize();
    cudaMemcpy(o.h, o.d, o.bytes, cudaMemcpyDeviceToHost);

    for (auto i = 0; i < o.len; i++) {
        if (i != 0 and i % 16 == 0) cout << "\n";
        printf("%3.0f ", o.h[i]);
    }

    return 0;
}
