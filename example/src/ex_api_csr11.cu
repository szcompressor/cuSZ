/**
 * @file ex_api_csr11.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-01-05
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "ex_common.cuh"

#include "utils.hh"
#include "wrapper/csr11.cuh"

using UNCOMPRESSED = float;
using COMPONENT    = cusz::CSR11<UNCOMPRESSED>;

void compress_time__alloc_inside(
    COMPONENT&     component,
    UNCOMPRESSED*& uncompressed,
    SIZE const     uncompressed_len,
    BYTE*&         compressed,
    SIZE&          compressed_len,
    cudaStream_t   stream)
{
    auto CR = [&]() { return 1.0 * uncompressed_len * 4 / compressed_len; };

    // one-time setup
    component.allocate_workspace(uncompressed_len);
    component.gather_new(uncompressed, uncompressed_len, compressed, compressed_len, stream, true /*debug*/);

    printf("(print in encoding)\n");
    printf("%-*s: %lu\n", 28, "compressed/subfile size", compressed_len);
    printf("%-*s: %.3lfx\n\n", 28, "compression ration (CR)", CR());
}

void decompress_time(COMPONENT& component, BYTE*& compressed, UNCOMPRESSED*& decompressed, cudaStream_t stream)
{
    component.scatter_new(compressed, decompressed, stream);
}

int main(int argc, char** argv)
{
    COMPONENT component;

    UNCOMPRESSED* uncompressed{nullptr};
    UNCOMPRESSED* decompressed{nullptr};
    SIZE          uncompressed_len = 3600 * 1800;
    // must be specially padded to a squire matrix
    uncompressed_len = Reinterpret1DTo2D::get_square_size(uncompressed_len);
    uncompressed_len *= uncompressed_len;

    BYTE*        compressed(nullptr);
    SIZE         compressed_len;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    // ----------------------------------------------------------------------------------------
    exp__prepare_data(&uncompressed, &decompressed, uncompressed_len, 0, 50, false);
    compress_time__alloc_inside(component, uncompressed, uncompressed_len, compressed, compressed_len, stream);
    // ----------------------------------------------------------------------------------------
    decompress_time(component, compressed, decompressed, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    verify(uncompressed, decompressed, uncompressed_len);
    // ----------------------------------------------------------------------------------------
    exp__free(uncompressed, decompressed);
    if (stream) CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}
