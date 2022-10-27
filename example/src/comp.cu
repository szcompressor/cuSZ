/**
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-10
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>

using std::cout;
using std::endl;

#include "cli/quality_viewer.hh"
#include "component/codec.hh"
#include "cusz.h"
#include "hf/hf_struct.h"
#include "kernel/cpplaunch_cuda.hh"
#include "kernel/launch_lossless.cuh"
#include "kernel/launch_spv.cuh"
#include "utils/cuda_err.cuh"
#include "utils/print.cuh"

namespace alpha {

struct {
    uint32_t x, y, z;
} len3;

typedef struct config {
    double eb;
    int    radius;
} config;

typedef struct file {
    // reserve for header
} file_t;

typedef struct header_superset {
    cusz_header header;
    int         nnz;
    size_t      total_nbit;
    size_t      total_ncell;
    int         hf_pardeg;
    int         hf_sublen;
} header_superset;

template <typename T, typename E, typename M>
struct runtime_data {
    T* h_data;

    T*        data;
    T*        outlier;
    uint32_t* outlier_idx;

    T* xdata;

    dim3   len3;
    size_t len;

    T*   anchor;
    dim3 anchor_len3;

    E*   errq;
    dim3 errq_len3;

    T* val;
    M* idx;
};

struct hf_set {
    hf_book* book_desc;
    // hf_chunk*     chunk_desc_d;
    // hf_chunk*     chunk_desc_h;
    hf_bitstream* bitstream_desc;

    uint8_t* revbook;
    size_t   revbook_nbyte;
    uint8_t* out;
    size_t   outlen;

    uint8_t* hl_comp;  // high level encoder API
    size_t   hl_complen;
};

template <typename H>
int get_revbook_nbyte(int booklen)
{
    return sizeof(H) * (2 * sizeof(H) * 8) + sizeof(H) * booklen;
}

}  // namespace alpha

template <typename T, typename E, typename FP, typename H = uint32_t, typename M = uint32_t>
cusz_error_status allocate_data(
    alpha::runtime_data<T, E, M>* d,
    dim3                          len3,
    alpha::hf_set*                hf,
    cusz::LosslessCodec<E, H, M>* codec,
    alpha::config*                config)
{
    auto   x   = len3.x;
    auto   y   = len3.y;
    auto   z   = len3.z;
    size_t len = x * y * z;

    // coarse-grained
    auto sublen = 768;
    auto pardeg = (len + sublen - 1) / sublen;
    cout << "pardeg\t" << pardeg << endl;
    hf->book_desc->booklen     = config->radius * 2;
    hf->bitstream_desc->sublen = sublen;
    hf->bitstream_desc->pardeg = pardeg;
    hf->revbook_nbyte          = alpha::get_revbook_nbyte<H>(hf->book_desc->booklen);

    CHECK_CUDA(cudaMallocHost(&d->h_data, sizeof(T) * len));
    CHECK_CUDA(cudaMalloc(&d->data, sizeof(T) * len));
    CHECK_CUDA(cudaMalloc(&d->outlier, sizeof(T) * len));
    CHECK_CUDA(cudaMalloc(&d->errq, sizeof(E) * len));

    CHECK_CUDA(cudaMalloc(&d->xdata, sizeof(T) * len));

    CHECK_CUDA(cudaMalloc(&d->val, sizeof(T) * len / 4));
    CHECK_CUDA(cudaMalloc(&d->idx, sizeof(uint32_t) * len / 4));

    codec->init(len, hf->book_desc->booklen, pardeg);

    // 22-10-12 LL API to be updated.

    // CHECK_CUDA(cudaMalloc(&hf->bitstream_desc->buffer, sizeof(H) * len));
    // CHECK_CUDA(cudaMalloc(&hf->bitstream_desc->bitstream, sizeof(T) * len / 2));  // check again
    // CHECK_CUDA(cudaMalloc(&hf->out, sizeof(H) * len / 4));
    // CHECK_CUDA(cudaMalloc(&hf->bitstream_desc->d_metadata->bits, sizeof(M) * pardeg));
    // CHECK_CUDA(cudaMalloc(&hf->bitstream_desc->d_metadata->cells, sizeof(M) * pardeg));
    // CHECK_CUDA(cudaMalloc(&hf->bitstream_desc->d_metadata->entries, sizeof(M) * pardeg));
    // CHECK_CUDA(cudaMallocHost(&hf->bitstream_desc->h_metadata->bits, sizeof(M) * pardeg));
    // CHECK_CUDA(cudaMallocHost(&hf->bitstream_desc->h_metadata->cells, sizeof(M) * pardeg));
    // CHECK_CUDA(cudaMallocHost(&hf->bitstream_desc->h_metadata->entries, sizeof(M) * pardeg));
    CHECK_CUDA(cudaMalloc(&hf->book_desc->freq, sizeof(uint32_t) * hf->book_desc->booklen));
    // CHECK_CUDA(cudaMalloc(&hf->book_desc->book, sizeof(H) * hf->book_desc->booklen));
    // CHECK_CUDA(cudaMalloc(&hf->revbook, hf->revbook_nbyte));
    CHECK_CUDA(cudaMalloc(&hf->hl_comp, 2 * len));

    // hf->d_metadata)
    return cusz_error_status::CUSZ_SUCCESS;
}

template <typename T, typename E, typename FP, typename H = uint32_t, typename M = uint32_t>
cusz_error_status deallocate_data(
    alpha::runtime_data<T, E, M>* d,
    alpha::hf_set*                hf,
    // cusz::LosslessCodec<E, H, M>* codec,
    alpha::config* config)
{
    CHECK_CUDA(cudaFreeHost(d->h_data));
    CHECK_CUDA(cudaFree(d->data));
    CHECK_CUDA(cudaFree(d->xdata));

    CHECK_CUDA(cudaFree(d->errq));

    CHECK_CUDA(cudaFree(d->val));
    CHECK_CUDA(cudaFree(d->idx));

    CHECK_CUDA(cudaFree(hf->book_desc->freq));
    CHECK_CUDA(cudaFree(hf->hl_comp));

    // hf->d_metadata)
    return cusz_error_status::CUSZ_SUCCESS;
}

template <typename T, typename E, typename FP, typename H = uint32_t, typename M = uint32_t>
cusz_error_status compressor(
    alpha::runtime_data<T, E, M>* data,
    alpha::hf_set*                hf,
    cusz::LosslessCodec<E, H, M>* codec,
    alpha::config*                config,
    alpha::header_superset*       header_st,
    bool                          use_proto,
    cudaStream_t                  stream)
{
    float time_pq = 0, time_hist = 0, time_spv = 0;
    // , time_book = 0, time_encoding = 0;

    if (not use_proto) {
        cout << "using optimized comp. kernel\n";
        cusz::cpplaunch_construct_LorenzoI<T, E, FP>(                                                   //
            data->data, data->len3, config->eb, config->radius,                                         //
            data->errq, data->len3, data->anchor, data->anchor_len3, data->outlier, data->outlier_idx,  //
            &time_pq, stream);
    }
    else {
        cout << "using prototype comp. kernel\n";
        cusz::cpplaunch_construct_LorenzoI_proto<T, E, FP>(                                             //
            data->data, data->len3, config->eb, config->radius,                                         //
            data->errq, data->len3, data->anchor, data->anchor_len3, data->outlier, data->outlier_idx,  //
            &time_pq, stream);
    }

    cout << "time-eq\t" << time_pq << endl;

    launch_spv_gather<T, M>(data->outlier, data->len, data->val, data->idx, header_st->nnz, time_spv, stream);

    cout << "time-spv\t" << time_spv << endl;
    cout << "nnz\t" << header_st->nnz << endl;

    cusz::cpplaunch_histogram<E>(
        data->errq, data->len, hf->book_desc->freq, hf->book_desc->booklen, &time_hist, stream);

    cout << "time-hist\t" << time_hist << endl;

    // 22-10-12 LL API temporarily not working

    // launch_gpu_parallel_build_codebook<E, H, M>(
    //     hf->book_desc->freq, (H*)hf->book_desc->book, hf->book_desc->booklen, hf->revbook, hf->revbook_nbyte,
    //     time_book, stream);

    // cout << "time-book\t" << time_book << endl;
    // 22-10-12 LL API temporarily not working

    // cusz::cpplaunch_coarse_grained_Huffman_encoding_rev1<E, H, M>(
    //     data->errq, data->len,              //
    //     hf->book_desc, hf->bitstream_desc,  //
    //     &hf->out, &hf->outlen, &time_encoding, stream);

    // cout << "time-encoding\t" << time_encoding << endl;

    codec->build_codebook(hf->book_desc->freq, hf->book_desc->booklen, stream);
    codec->encode(data->errq, data->len, hf->hl_comp, hf->hl_complen, stream);

    // header_st->total_nbit = std::accumulate(
    //     (M*)hf->bitstream_desc->h_metadata->bits, (M*)hf->bitstream_desc->h_metadata->bits +
    //     hf->bitstream_desc->pardeg, (size_t)0);
    // header_st->total_ncell = std::accumulate(
    //     (M*)hf->bitstream_desc->h_metadata->cells,
    //     (M*)hf->bitstream_desc->h_metadata->cells + hf->bitstream_desc->pardeg, (size_t)0);

    // cout << "total nbit\t" << header_st->total_nbit << endl;
    // cout << "total ncell\t" << header_st->total_ncell << endl;

    return cusz_error_status::CUSZ_SUCCESS;
}

template <typename T, typename E, typename FP, typename H = uint32_t, typename M = uint32_t>
cusz_error_status decompressor(
    alpha::runtime_data<T, E, M>* data,
    alpha::hf_set*                hf,
    cusz::LosslessCodec<E, H, M>* codec,
    alpha::header_superset*       header_st,
    bool                          use_proto,
    cudaStream_t                  stream)
{
    float time_scatter = 0,
          // time_hf = 0,
        time_d_pq = 0;

    launch_spv_scatter<T, uint32_t>(data->val, data->idx, header_st->nnz, data->xdata, time_scatter, stream);
    cout << "decomp-time-spv\t" << time_scatter << endl;

    codec->decode(hf->hl_comp, data->errq);

    // 22-10-12 LL API temporarily not working
    // cusz::cpplaunch_coarse_grained_Huffman_decoding<E, H, M>(
    //     (H*)hf->bitstream_desc->bitstream, hf->revbook, hf->revbook_nbyte, (M*)hf->bitstream_desc->d_metadata->bits,
    //     (M*)hf->bitstream_desc->d_metadata->entries, header_st->hf_sublen, header_st->hf_pardeg, data->errq,
    //     &time_hf, stream);

    // cout << "decomp-time-hf\t" << time_hf << endl;

    if (not use_proto) {
        cout << "using optimized comp. kernel\n";
        cusz::cpplaunch_reconstruct_LorenzoI<T, E, FP>(                                                 //
            data->errq, data->len3, data->anchor, data->anchor_len3, data->outlier, data->outlier_idx,  // input
            header_st->header.eb, header_st->header.radius,  // input (config)
            data->xdata, data->len3,                         // output
            &time_d_pq, stream);
    }
    else {
        cout << "using prototype comp. kernel\n";
        cusz::cpplaunch_reconstruct_LorenzoI_proto<T, E, FP>(                                           //
            data->errq, data->len3, data->anchor, data->anchor_len3, data->outlier, data->outlier_idx,  // input
            header_st->header.eb, header_st->header.radius,  // input (config)
            data->xdata, data->len3,                         // output
            &time_d_pq, stream);
    }

    cout << "decomp-time-pq\t" << time_d_pq << endl;

    return cusz_error_status::CUSZ_SUCCESS;
}

template <typename T, typename E>
void f(
    std::string&            fname,
    dim3                    len3,
    alpha::hf_set*          hf,
    alpha::header_superset* header_st,
    alpha::config*          config,
    bool                    use_proto)
{
    using FP = T;
    using M  = uint32_t;
    using H  = uint32_t;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto data  = new alpha::runtime_data<T, E, M>;
    data->len3 = len3;
    data->len  = len3.x * len3.y * len3.z;

    cusz::LosslessCodec<E, uint32_t, uint32_t> codec;

    allocate_data<T, E, FP, H, M>(data, len3, hf, &codec, config);

    io::read_binary_to_array(fname, data->h_data, data->len);
    CHECK_CUDA(cudaMemcpy(data->data, data->h_data, sizeof(T) * data->len, cudaMemcpyHostToDevice));

    compressor<T, E, FP, H, M>(data, hf, &codec, config, header_st, use_proto, stream);

    decompressor<T, E, FP, H, M>(data, hf, &codec, header_st, use_proto, stream);

    /* view quality */ cusz::QualityViewer::echo_metric_gpu(data->xdata, data->data, data->len);

    deallocate_data<T, E, FP, H, M>(data, hf, config);

    cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
    if (argc < 6) {
        printf("0    1             2     3 4 5 6  [7]     [8:128]  [9:yes]\n");
        printf("PROG /path/to/file DType X Y Z EB [EType] [Radius] [Use Prototype]\n");
        printf(" 2  DType: \"F\" for `float`, \"D\" for `double`\n");
        printf("[7] EType: \"ui{8,16,32}\" for `uint{8,16,32}_t` as quant-code type\n");
        exit(0);
    }

    auto fname = std::string(argv[1]);
    auto dtype = std::string(argv[2]);
    auto x     = atoi(argv[3]);
    auto y     = atoi(argv[4]);
    auto z     = atoi(argv[5]);
    auto eb    = atof(argv[6]);

    std::string etype;
    if (argc > 7)
        etype = std::string(argv[7]);
    else
        etype = "ui16";

    int radius;
    if (argc > 8)
        radius = atoi(argv[8]);
    else
        radius = 128;

    bool use_proto;
    if (argc > 9)
        use_proto = std::string(argv[9]) == "yes";
    else
        use_proto = false;

    using T = float;
    using E = uint16_t;

    auto len3 = dim3(x, y, z);

    auto header_st           = new alpha::header_superset;
    header_st->header.eb     = eb;
    header_st->header.radius = radius;

    auto config = new alpha::config{eb, radius};

    auto hf       = new alpha::hf_set;
    hf->book_desc = new hf_book;
    // 22-10-12 LL API temporarily not working
    hf->bitstream_desc = new hf_bitstream;
    // hf->bitstream_desc->d_metadata = new hf_chunk;
    // hf->bitstream_desc->h_metadata = new hf_chunk;

    //// dispatch

    auto radius_legal = [&](int const sizeof_T) {
        size_t upper_bound = 1lu << (sizeof_T * 8);
        cout << upper_bound << endl;
        cout << radius * 2 << endl;
        if ((radius * 2) > upper_bound) throw std::runtime_error("Radius overflows error-quantization type.");
    };

    if (dtype == "F") {
        if (etype == "ui8") {
            radius_legal(1);
            f<float, uint8_t>(fname, len3, hf, header_st, config, use_proto);
        }
        else if (etype == "ui16") {
            radius_legal(2);
            f<float, uint16_t>(fname, len3, hf, header_st, config, use_proto);
        }
        else if (etype == "ui32") {
            radius_legal(4);
            f<float, uint32_t>(fname, len3, hf, header_st, config, use_proto);
        }
        // else if (etype == "fp32") {
        //     f<float, float>(fname, len3, hf, header_st, config, use_proto);
        // }
    }
    else if (dtype == "D") {
        if (etype == "ui8") {
            radius_legal(1);
            f<double, uint8_t>(fname, len3, hf, header_st, config, use_proto);
        }
        else if (etype == "ui16") {
            radius_legal(2);
            f<double, uint16_t>(fname, len3, hf, header_st, config, use_proto);
        }
        else if (etype == "ui32") {
            radius_legal(4);
            f<double, uint32_t>(fname, len3, hf, header_st, config, use_proto);
        }
        // else if (etype == "fp32") {
        //     f<double, float>(fname, len3, hf, header_st, config, use_proto);
        // }
    }
    else
        throw std::runtime_error("not a valid dtype.");

    return 0;
}