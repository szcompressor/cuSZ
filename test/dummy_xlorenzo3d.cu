
#include "../src/kernel/lorenzo.h"
#include "../src/metadata.hh"

using Data  = float;
using Quant = unsigned short;

int main()
{
    LorenzoUnzipContext ctx;
    ctx.d0      = 512;
    ctx.d1      = 512;
    ctx.d2      = 512;
    ctx.stride1 = 512;
    ctx.stride2 = 512 * 512;
    ctx.radius  = 0;
    ctx.ebx2    = 1;

    auto length = ctx.d0 * ctx.d1 * ctx.d2;

    union {
        Data* data;
        Data* outlier;
    } array;
    unsigned short* quant;

    cudaMallocManaged(&array.data, length * sizeof(Data));
    cudaMallocManaged(&quant, length * sizeof(Quant));

    kernel::x_lorenzo_3d1l_8x8x8_v3<Data, Quant>
        <<<dim3(ctx.d0 / 8, ctx.d1 / 8, ctx.d2 / 8), dim3(8, 1, 8)>>>(ctx, array.data, array.outlier, quant);

    cudaDeviceSynchronize();

    return 0;
}
