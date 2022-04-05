/**
 * @file compressor.cu
 * @author Jiannan Tian
 * @brief cuSZ compressor of the default path
 * @version 0.3
 * @date 2021-10-05
 * (create) 2020-02-12; (release) 2020-09-20;
 * (rev.1) 2021-01-16; (rev.2) 2021-07-12; (rev.3) 2021-09-06; (rev.4) 2021-10-05
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "compressor.cuh"

/*
#define DPCOMPRESSOR_TYPE template <class BINDING>
#define DPCOMPRESSOR DefaultPathCompressor<BINDING>

DPCOMPRESSOR_TYPE
DPCOMPRESSOR& DPCOMPRESSOR::analyze_compressibility()
{
    if (this->ctx->report.compressibility) {
        this->freq.template alloc<kHOST>().device2host();
        book.template alloc<kHOST>().device2host();

        Analyzer analyzer{};
        analyzer  //
            .estimate_compressibility_from_histogram(this->freq.hptr, this->dict_size)
            .template get_stat_from_huffman_book<H>(this->freq.hptr, book.hptr, this->ctx->data_len, this->dict_size)
            .print_compressibility(true);

        cudaFreeHost(this->freq.hptr);
        cudaFreeHost(book.hptr);
    }

    return *this;
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR& DPCOMPRESSOR::internal_eval_try_export_book()
{
    // internal evaluation, not stored in sz archive
    if (this->ctx->export_raw.book) {
        cudaMallocHost(&book.hptr, this->dict_size * sizeof(decltype(book.hptr)));
        book.device2host();

        std::stringstream s;
        s << this->ctx->fname.path_basename + "-" << this->dict_size << "-ui" << sizeof(H) << ".lean-book";

        // TODO as part of dump
        io::write_array_to_binary(s.str(), book.hptr, this->dict_size);

        cudaFreeHost(book.hptr);
        book.hptr = nullptr;

        LOGGING(LOG_INFO, "exporting codebook as binary; suffix: \".lean-book\"");

        this->dataseg.nbyte.at(cusz::SEG::BOOK) = this->dict_size * sizeof(H);
    }
    return *this;
}

DPCOMPRESSOR_TYPE DPCOMPRESSOR& DPCOMPRESSOR::internal_eval_try_export_quant()
{
    // internal_eval
    if (this->ctx->export_raw.quant) {  //
        this->quant.template alloc<kHOST>();
        this->quant.device2host();

        this->dataseg.nbyte.at(cusz::SEG::QUANT) = this->quant.nbyte();

        // TODO as part of dump
        io::write_array_to_binary(
            this->ctx->fname.path_basename + ".lean-this->quant", this->quant.hptr,
            BINDING::template get_len_uncompressed(predictor, codec));
        LOGGING(LOG_INFO, "exporting this->quant as binary; suffix: \".lean-this->quant\"");
        LOGGING(LOG_INFO, "exiting");
        exit(0);
    }
    return *this;
}

DPCOMPRESSOR_TYPE DPCOMPRESSOR& DPCOMPRESSOR::try_skip_huffman()
{
    // decide if skipping Huffman coding
    if (this->ctx->to_skip.huffman) {
        // cudaMallocHost(&this->quant.hptr, this->quant.nbyte());
        this->quant  //
            .template alloc<kHOST>()
            .device2host();

        // TODO: as part of cusza
        io::write_array_to_binary(
            this->ctx->fname.path_basename + ".this->quant", this->quant.hptr,
            BINDING::template get_len_uncompressed(predictor, codec));
        LOGGING(LOG_INFO, "to store this->quant.code directly (Huffman enc skipped)");
        exit(0);
    }

    return *this;
}
*/

// TODO the experiments left out
/*
this->analyze_compressibility()  //
    .internal_eval_try_export_book()
    .internal_eval_try_export_quant();
*/

#define F32_DEFAULT_PATH_COMPRESSOR DefaultPathCompressor<DefaultPath<float>::DefaultBinding>

// template class DefaultPathCompressor<DefaultPath<float>::DefaultBinding>;
template class F32_DEFAULT_PATH_COMPRESSOR;
template void F32_DEFAULT_PATH_COMPRESSOR::init<cuszCTX>(cuszCTX*, bool);
template void F32_DEFAULT_PATH_COMPRESSOR::init<cuszHEADER>(cuszHEADER*, bool);
