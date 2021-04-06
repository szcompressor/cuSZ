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

int main()
{
    struct data_t<float> o;
    o.len   = 512;
    o.bytes = o.len * sizeof(decltype(o)::Type);

    struct data_t<uint16_t> q;
    q.len   = 512;
    q.bytes = q.len * sizeof(decltype(q)::Type);

    cudaMallocHost(&o.h, o.bytes);
    cudaMalloc(&o.d, o.bytes);

    cudaMallocHost(&q.h, q.bytes);
    cudaMalloc(&q.d, q.bytes);

    for (auto i = 0; i < o.len; i++) { q.h[i] = 1, o.h[i] = 0; }

    cudaMemcpy(o.d, o.h, o.bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(q.d, q.h, q.bytes, cudaMemcpyHostToDevice);

    LorenzoUnzipContext ctx;
    ctx.d0      = 8;
    ctx.d1      = 8;
    ctx.d2      = 8;
    ctx.stride1 = 8;
    ctx.stride2 = 8 * 8;
    ctx.radius  = 0;
    ctx.ebx2    = 1;

    cusz::predictor_quantizer::x_lorenzo_3d1l_8x8x8_v0_v2  //
        <decltype(o)::Type, decltype(q)::Type>          //
        <<<dim3(1, 1, 1), dim3(8, 2, 8)>>>(ctx, o.d, o.d, q.d);

    cudaDeviceSynchronize();
    cudaMemcpy(o.h, o.d, o.bytes, cudaMemcpyDeviceToHost);

    for (auto i = 0; i < o.len; i++) {
        if (i != 0 and i % 8 == 0) cout << "\n";
        if (i != 0 and i % 64 == 0) cout << "\n";
        printf("%3.0f ", o.h[i]);
    }

    return 0;
}
