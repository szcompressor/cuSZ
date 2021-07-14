#include "../src/kernel/lorenzo.h"
#include "../src/metadata.hh"

using Data  = float;
using Quant = unsigned short;

int main()
{
    LorenzoUnzipContext ctx;
    ctx.d0      = 3600;
    ctx.d1      = 1800;
    ctx.stride1 = 3600;
    ctx.radius  = 0;
    ctx.ebx2    = 1;

    auto length = ctx.d0 * ctx.d1;

    union {
        Data* data;
        Data* outlier;
    } array;
    unsigned short* quant;

    cudaMallocManaged(&array.data, length * sizeof(Data));
    cudaMallocManaged(&quant, length * sizeof(Quant));

    cusz::x_lorenzo_2d1l_16x16_v1<Data, Quant>
        <<<dim3(ctx.d0 / 16, ctx.d1 / 16), dim3(16, 2)>>>(ctx, array.data, array.outlier, quant);
    cudaDeviceSynchronize();

    return 0;
}
