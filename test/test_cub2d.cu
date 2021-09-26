#if CUDART_VERSION >= 11000
// #pragma message __FILE__ ": (CUDA 11 onward), cub from system path"
#include <cub/cub.cuh>
#else
// #pragma message __FILE__ ": (CUDA 10 or earlier), cub from git submodule"
#include "../external/cub/cub/cub.cuh"
#endif

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>
using std::cout;
using std::endl;
//#include "../src/dualquant.cuh"
#include "../src/metadata.hh"
#include "../src/common.hh"

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

static const auto ProblemSize   = 8;
static const auto MultiBlock    = 1;
static const auto Sequentiality = 2;  // items per thread

template <typename Data, typename Quant>
__global__ void cub2d(lorenzo_unzip ctx, Data* xdata, Data* outlier, Quant* quant)
{
    static const auto Block = ProblemSize;

    static const auto block_dim = Block / Sequentiality;  // dividable

    // coalesce-load (warp-striped) and transpose in shmem (similar for store)
    typedef cub::BlockLoad<Data, block_dim, Sequentiality, cub::BLOCK_LOAD_WARP_TRANSPOSE, Block>   BlockLoadT_outlier;
    typedef cub::BlockLoad<Quant, block_dim, Sequentiality, cub::BLOCK_LOAD_WARP_TRANSPOSE, Block>  BlockLoadT_quant;
    typedef cub::BlockStore<Data, block_dim, Sequentiality, cub::BLOCK_STORE_WARP_TRANSPOSE, Block> BlockStoreT_xdata;
    typedef cub::BlockScan<Data, block_dim, cub::BLOCK_SCAN_RAKING_MEMOIZE, Block>                  BlockScanT_xdata;

    __shared__ union TempStorage {  // overlap shared memory space
        typename BlockLoadT_outlier::TempStorage load_outlier;
        typename BlockLoadT_quant::TempStorage   load_quant;
        typename BlockStoreT_xdata::TempStorage  store_xdata;
        typename BlockScanT_xdata::TempStorage   scan_xdata;
    } temp_storage;

    // thread-scope tiled data
    union ThreadData {
        Data xdata[Sequentiality];
        Data outlier[Sequentiality];
    } thread_scope;
    Quant thread_scope_quant[Sequentiality];

    // TODO pad for potential out-of-range access
    // (bix * bdx * Sequentiality) denotes the start of the data chunk that belongs to this thread block
    BlockLoadT_quant(temp_storage.load_quant).Load(quant + (bix * bdx) * Sequentiality, thread_scope_quant);
    __syncthreads();  // barrier for shmem reuse
    BlockLoadT_outlier(temp_storage.load_outlier).Load(outlier + (bix * bdx) * Sequentiality, thread_scope.outlier);
    __syncthreads();  // barrier for shmem reuse

    auto radius = static_cast<Data>(ctx.radius);
#pragma unroll
    for (auto i = 0; i < Sequentiality; i++) {
        auto id = (bix * bdx + tix) * Sequentiality + i;
        thread_scope.xdata[i] =
            id < ctx.d0 ? thread_scope.outlier[i] + static_cast<Data>(thread_scope_quant[i]) - radius : 0;
    }
    __syncthreads();

    BlockScanT_xdata(temp_storage.scan_xdata).InclusiveSum(thread_scope.xdata, thread_scope.xdata);
    __syncthreads();  // barrier for shmem reuse

#pragma unroll
    for (auto i = 0; i < Sequentiality; i++) thread_scope.xdata[i] *= ctx.ebx2;
    __syncthreads();  // barrier for shmem reuse

    BlockStoreT_xdata(temp_storage.store_xdata).Store(xdata + (bix * bdx) * Sequentiality, thread_scope.xdata);
}

int main()
{
    struct data_t<float> o;
    o.len   = ProblemSize * ProblemSize * MultiBlock;
    o.bytes = o.len * sizeof(decltype(o)::Type);

    struct data_t<uint16_t> q;
    q.len   = ProblemSize * ProblemSize * MultiBlock;
    q.bytes = q.len * sizeof(decltype(q)::Type);

    cudaMallocHost(&o.h, o.bytes);
    cudaMalloc(&o.d, o.bytes);

    cudaMallocHost(&q.h, q.bytes);
    cudaMalloc(&q.d, q.bytes);

    for (auto i = 0; i < o.len; i++) { q.h[i] = 1, o.h[i] = 0; }

    cudaMemcpy(o.d, o.h, o.bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(q.d, q.h, q.bytes, cudaMemcpyHostToDevice);

    LorenzoUnzipContext ctx;
    ctx.d0 = ProblemSize;
    //    ctx.d1      = 16 * 2;
    ctx.d1      = ProblemSize;
    ctx.stride1 = ProblemSize;
    ctx.radius  = 0;
    ctx.ebx2    = 1;

    cub2d<<<dim3(1, 1), dim3(ProblemSize / Sequentiality, ProblemSize)>>>(ctx, o.d, o.d, q.d);

    cudaDeviceSynchronize();
    cudaMemcpy(o.h, o.d, o.bytes, cudaMemcpyDeviceToHost);

    for (auto i = 0; i < o.len; i++) {
        if (i != 0 and i % ProblemSize == 0) cout << "\n";
        printf("%3.0f ", o.h[i]);
    }

    return 0;
}
