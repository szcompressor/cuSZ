/**
 * @file ex_api_huffcoarse.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-01-03
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "ex_common.cuh"

#include "utils.hh"
#include "wrapper/huffman_coarse.cuh"

using UNCOMPRESSED = uint16_t;
using COMPONENT    = cusz::HuffmanCoarse<UNCOMPRESSED, uint32_t, uint32_t>;

int const booklen          = 1024;
int const sublen           = 4096;
SIZE      uncompressed_len = 3600 * 1800;
int       pardeg           = ConfigHelper::get_npart(uncompressed_len, sublen);

void compress_time__alloc_inside(
    COMPONENT&     component,
    UNCOMPRESSED*& uncompressed,
    SIZE const     uncompressed_len,
    BYTE*&         compressed,
    SIZE&          compressed_len,
    cudaStream_t   stream)
{
    auto CR          = [&]() { return 1.0 * uncompressed_len * 4 / compressed_len; };
    auto peek_header = [&]() {
        auto header = new COMPONENT::HEADER;
        cudaMemcpy(header, compressed, sizeof(COMPONENT::HEADER), cudaMemcpyDeviceToHost);

        printf("header::%-*s: %d\n", 20, "header_nbyte", (*header).header_nbyte);
        printf("header::%-*s: %d\n", 20, "booklen", (*header).booklen);
        printf("header::%-*s: %d\n", 20, "sublen", (*header).sublen);
        printf("header::%-*s: %lu\n", 20, "uncompressed_len", (*header).uncompressed_len);
        printf("header::%-*s: %lu\n", 20, "total_nbit", (*header).total_nbit);
        printf("header::%-*s: %lu\n", 20, "total_ncell", (*header).total_ncell);

        printf("\n");

        PRINT_HEADER_ENTRY(HEADER)
        PRINT_HEADER_ENTRY(REVBOOK)
        PRINT_HEADER_ENTRY(PAR_NBIT)
        PRINT_HEADER_ENTRY(PAR_ENTRY)
        PRINT_HEADER_ENTRY(BITSTREAM)
        PRINT_HEADER_ENTRY(END)

        delete header;
    };

    // one-time setup
    component.allocate_workspace(uncompressed_len, booklen, pardeg);
    component.encode_new(uncompressed, uncompressed_len, booklen, sublen, compressed, compressed_len, stream);

    printf("(print in encoding)\n");
    printf("%-*s: %lu\n", 28, "compressed/subfile size", compressed_len);
    printf("%-*s: %.3lfx\n\n", 28, "compression ration (CR)", CR());
    peek_header();
}

void decompress_time(COMPONENT& component, BYTE*& compressed, UNCOMPRESSED*& decompressed, cudaStream_t stream)
{
    component.decode_new(compressed, decompressed, stream);
}

int main(int argc, char** argv)
{
    COMPONENT component;

    UNCOMPRESSED* uncompressed{nullptr};
    UNCOMPRESSED* decompressed{nullptr};
    BYTE*         compressed(nullptr);
    SIZE          compressed_len;
    cudaStream_t  stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    // ----------------------------------------------------------------------------------------
    exp__prepare_data(&uncompressed, &decompressed, uncompressed_len, 512, 20, false);
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
