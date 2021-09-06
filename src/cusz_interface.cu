/**
 * @file cusz_workflow.cu
 * @author Jiannan Tian
 * @brief Workflow of cuSZ.
 * @version 0.3
 * @date 2021-07-12
 * (create) 2020-02-12; (release) 2020-09-20; (rev.1) 2021-01-16; (rev.2) 2021-07-12
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cxxabi.h>
#include <bitset>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "analysis/analyzer.hh"
#include "argparse.hh"
#include "cusz_interface.h"
#include "kernel/dryrun.h"
#include "kernel/lorenzo.h"
#include "metadata.hh"
#include "type_trait.hh"
#include "utils.hh"
#include "wrapper/extrap_lorenzo.h"
#include "wrapper/handle_sparsity.h"
#include "wrapper/huffman_enc_dec.cuh"
#include "wrapper/huffman_parbook.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

#define DATATYPE struct PartialData<typename DataTrait<If_FP, DataByte>::Data>

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_compress(argpack* ap, DATATYPE* in_data, dim3 xyz, metadata_pack* mp, unsigned int optional_w)
{
    using Data  = typename DataTrait<If_FP, DataByte>::Data;
    using Quant = typename QuantTrait<QuantByte>::Quant;
    using Huff  = typename HuffTrait<HuffByte>::Huff;

    Compressor<Data, Quant, Huff, float> cuszc(ap, ap->len, ap->eb);

    cuszc.lorenzo_dryrun(in_data);  // subject to change

    struct PartialData<Quant> quant(ap->len + HuffConfig::Db_encode);
    cudaMalloc(&quant.dptr, quant.nbyte());

    struct PartialData<unsigned int> freq(ap->dict_size);
    cudaMalloc(&freq.dptr, freq.nbyte());

    struct PartialData<Huff> book(ap->dict_size);
    cudaMalloc(&book.dptr, book.nbyte()), book.memset(0xff);

    struct PartialData<uint8_t> revbook(cuszc.get_revbook_nbyte());
    cudaMalloc(&revbook.dptr, revbook.nbyte());
    cudaMallocHost(&revbook.hptr, revbook.nbyte());  // to write to disk later

    cuszc  //
        .predict_quantize(in_data, xyz, &quant)
        .gather_outlier(in_data)
        .try_skip_huffman(&quant);

    // release in_data; subject to change
    cudaFree(in_data->dptr);

    cuszc.get_freq_and_codebook(&quant, &freq, &book, &revbook)
        .analyze_compressibility(&freq, &book)
        .internal_eval_try_export_book(&book)
        .internal_eval_try_export_quant(&quant)
        .export_revbook(&revbook)
        .huffman_encode(&quant, &book)
        .try_report_time()
        .pack_metadata(mp);

    cudaFree(quant.dptr), cudaFree(freq.dptr), cudaFree(book.dptr), cudaFree(revbook.dptr);
}

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_decompress(argpack* ap, metadata_pack* mp)
{
    using Data  = typename DataTrait<If_FP, DataByte>::Data;
    using Quant = typename QuantTrait<QuantByte>::Quant;
    using Huff  = typename HuffTrait<HuffByte>::Huff;

    Decompressor<Data, Quant, Huff, float> cuszd(mp, ap);

    auto xyz = dim3(ap->dim4._0, ap->dim4._1, ap->dim4._2);

    struct PartialData<Quant> quant(cuszd.length.quant);
    cudaMalloc(&quant.dptr, quant.nbyte());
    cudaMallocHost(&quant.hptr, quant.nbyte());

    struct PartialData<Data> _data(cuszd.mxm + MetadataTrait<1>::Block);  // TODO ad hoc size
    cudaMalloc(&_data.dptr, _data.nbyte());
    cudaMallocHost(&_data.hptr, _data.nbyte());
    auto xdata = _data.dptr, outlier = _data.dptr;

    cuszd.huffman_decode(&quant)
        .scatter_outlier(outlier)
        .reversed_predict_quantize(xdata, quant.dptr, xyz)
        .try_report_time();

    // copy decompressed data to host
    _data.d2h();

    cuszd
        .calculate_archive_nbyte()  //
        .try_compare(_data.hptr)
        .try_write2disk(_data.hptr);
}

#define CUSZ_COMPRESS(DBYTE, QBYTE, HBYTE)                  \
    template void cusz_compress<true, DBYTE, QBYTE, HBYTE>( \
        argpack*, struct PartialData<float>*, dim3, metadata_pack*, unsigned int);

CUSZ_COMPRESS(4, 1, 4)
CUSZ_COMPRESS(4, 1, 8)
CUSZ_COMPRESS(4, 2, 4)
CUSZ_COMPRESS(4, 2, 8)

#define CUSZ_DECOMPRESS(DBYTE, QBYTE, HBYTE) \
    template void cusz_decompress<true, DBYTE, QBYTE, HBYTE>(argpack*, metadata_pack*);

CUSZ_DECOMPRESS(4, 1, 4)
CUSZ_DECOMPRESS(4, 1, 8)
CUSZ_DECOMPRESS(4, 2, 4)
CUSZ_DECOMPRESS(4, 2, 8)
