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

auto demangle = [](const char* name) -> string {
    int   status = -4;
    char* res    = abi::__cxa_demangle(name, nullptr, nullptr, &status);

    const char* const demangled_name = (status == 0) ? res : name;
    string            ret_val(demangled_name);
    free(res);
    return ret_val;
};

namespace {

template <typename Data>
void report_decompression_time(size_t len, float lossy, float outlier, float lossless)
{
    auto display_throughput = [](float time, size_t nbyte) {
        auto throughput = nbyte * 1.0 / (1024 * 1024 * 1024) / (time * 1e-3);
        cout << throughput << "GiB/s\n";
    };
    //
    cout << "\nTIME in milliseconds\t================================================================\n";
    float all = lossy + outlier + lossless;

    printf("TIME\tscatter outlier:\t%f\t", outlier), display_throughput(outlier, len * sizeof(Data));
    printf("TIME\tHuffman decode:\t%f\t", lossless), display_throughput(lossless, len * sizeof(Data));
    printf("TIME\treconstruct:\t%f\t", lossy), display_throughput(lossy, len * sizeof(Data));

    cout << "TIME\t--------------------------------------------------------------------------------\n";

    printf("TIME\tdecompress (sum):\t%f\t", all), display_throughput(all, len * sizeof(Data));

    cout << "TIME\t================================================================================\n\n";
}

void PackMetadata(argpack* ap, metadata_pack* mp, const int nnz)
{
    mp->dim4    = ap->dim4;
    mp->stride4 = ap->stride4;
    mp->nblk4   = ap->nblk4;
    mp->ndim    = ap->ndim;
    mp->eb      = ap->eb;
    mp->len     = ap->len;

    mp->nnz = nnz;

    if (ap->dtype == "f32") mp->dtype = DataType::kF32;
    if (ap->dtype == "f64") mp->dtype = DataType::kF64;

    mp->quant_byte    = ap->quant_byte;
    mp->huff_byte     = ap->huff_byte;
    mp->huffman_chunk = ap->huffman_chunk;
    mp->skip_huffman  = ap->sz_workflow.skip_huffman;
}

void UnpackMetadata(argpack* ap, metadata_pack* mp, int& nnz)
{
    ap->dim4    = mp->dim4;
    ap->stride4 = mp->stride4;
    ap->nblk4   = mp->nblk4;
    ap->ndim    = mp->ndim;
    ap->eb      = mp->eb;
    ap->len     = mp->len;

    nnz = mp->nnz;

    if (mp->dtype == DataType::kF32) ap->dtype = "f32";
    if (mp->dtype == DataType::kF64) ap->dtype = "f64";

    ap->quant_byte               = mp->quant_byte;
    ap->huff_byte                = mp->huff_byte;
    ap->huffman_chunk            = mp->huffman_chunk;
    ap->sz_workflow.skip_huffman = mp->skip_huffman;
}

}  // namespace

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
        .try_report_time();

    cudaFree(quant.dptr), cudaFree(freq.dptr), cudaFree(book.dptr), cudaFree(revbook.dptr);

    PackMetadata(ap, mp, cuszc.length.nnz_outlier);
    mp->num_bits      = cuszc.huffman_meta.num_bits;
    mp->num_uints     = cuszc.huffman_meta.num_uints;
    mp->revbook_nbyte = cuszc.huffman_meta.revbook_nbyte;
}

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_decompress(argpack* ap, metadata_pack* mp)
{
    using Data  = typename DataTrait<If_FP, DataByte>::Data;
    using Quant = typename QuantTrait<QuantByte>::Quant;
    using Huff  = typename HuffTrait<HuffByte>::Huff;

    int nnz_outlier = 0;
    UnpackMetadata(ap, mp, nnz_outlier);
    auto num_uints     = mp->num_uints;
    auto revbook_nbyte = mp->revbook_nbyte;

    auto xyz = dim3(ap->dim4._0, ap->dim4._1, ap->dim4._2);

    float time_lossy{0}, time_outlier{0}, time_lossless{0};

    auto  len      = ap->len;
    auto& workflow = ap->sz_workflow;
    auto& subfiles = ap->subfiles;

    auto m   = static_cast<size_t>(ceil(sqrt(ap->len)));
    auto mxm = m * m;

    auto radius = ap->radius;
    auto eb     = ap->eb;
    auto ebx2   = (eb * 2);

    logging(log_info, "invoke lossy-reconstruction");

    struct PartialData<Quant> quant(len);
    cudaMalloc(&quant.dptr, quant.nbyte());
    cudaMallocHost(&quant.hptr, quant.nbyte());

    struct PartialData<Data> _data(mxm + MetadataTrait<1>::Block);  // TODO ad hoc size
    cudaMalloc(&_data.dptr, _data.nbyte());
    cudaMallocHost(&_data.hptr, _data.nbyte());
    auto xdata   = _data.dptr;
    auto outlier = _data.dptr;

    // step 1: read from filesystem or do Huffman decoding to get quant code
    if (workflow.skip_huffman) {
        logging(log_info, "load quant.code from filesystem");
        io::read_binary_to_array(subfiles.decompress.in_quant, quant.hptr, quant.len);
        quant.h2d();
    }
    else {
        logging(log_info, "Huffman decode -> quant.code");
        lossless::HuffmanDecode<Quant, Huff>(
            subfiles.path2file, &quant, ap->len, ap->huffman_chunk, num_uints, ap->dict_size, time_lossless);
    }

    {
        OutlierHandler<Data> csr(ap->len, nnz_outlier);

        uint8_t *h_csr_file, *d_csr_file;
        cudaMallocHost((void**)&h_csr_file, csr.bytelen.total);
        cudaMalloc((void**)&d_csr_file, csr.bytelen.total);

        io::read_binary_to_array<uint8_t>(subfiles.decompress.in_outlier, h_csr_file, csr.bytelen.total);
        cudaMemcpy(d_csr_file, h_csr_file, csr.bytelen.total, cudaMemcpyHostToDevice);

        csr.extract(d_csr_file).scatter_CUDA10(outlier, time_outlier);
    }

    /********************************************************************************
     * lorenzo reconstruction
     ********************************************************************************/
    if (workflow.predictor == "lorenzo") {
        decompress_lorenzo_reconstruct(xdata, quant.dptr, xyz, ap->ndim, eb, radius, time_lossy);
    }
    else if (workflow.predictor == "spline3d") {
        throw std::runtime_error("spline not impl'ed");
        if (ap->ndim != 3) throw std::runtime_error("Spline3D must be for 3D data.");
        // decompress_spline3d_reconstruct(xdata, quant.dptr, xyz, ap->ndim, eb, radius, time_lossy);
    }
    else {
        throw std::runtime_error("need to specify predcitor");
    }

    /********************************************************************************
     * report time
     ********************************************************************************/
    if (ap->report.time) report_decompression_time<Data>(len, time_lossy, time_outlier, time_lossless);

    // copy decompressed data to host
    _data.d2h();

    logging(log_info, "reconstruct error-bounded datum");

    size_t archive_bytes = 0;
    // TODO huffman chunking metadata
    if (not workflow.skip_huffman)
        archive_bytes += num_uints * sizeof(Huff)  // Huffman coded
                         + revbook_nbyte;          // chunking metadata and reverse codebook
    else
        archive_bytes += ap->len * sizeof(Quant);
    archive_bytes += nnz_outlier * (sizeof(Data) + sizeof(int)) + (m + 1) * sizeof(int);

    if (workflow.skip_huffman) {
        logging(
            log_info, "dtype is \"", demangle(typeid(Data).name()), "\", and quant. code type is \"",
            demangle(typeid(Quant).name()), "\"; a CR of no greater than ", (sizeof(Data) / sizeof(Quant)),
            " is expected when Huffman codec is skipped.");
    }

    if (workflow.pre_binning) logging(log_info, "Because of 2x2->1 binning, extra 4x CR is added.");

    // TODO move CR out of verify_data
    if (not subfiles.decompress.in_origin.empty() and ap->report.quality) {
        logging(log_info, "load the original datum for comparison");

        auto odata = io::read_binary_to_new_array<Data>(subfiles.decompress.in_origin, len);

        analysis::verify_data(&ap->stat, _data.hptr, odata, len);
        analysis::print_data_quality_metrics<Data>(
            &ap->stat, false, ap->eb, archive_bytes, workflow.pre_binning ? 4 : 1, true);
    }
    logging(log_info, "output:", subfiles.path2file + ".szx");

    if (workflow.skip_write2disk)
        logging(log_dbg, "skip writing unzipped to filesystem");
    else {
        io::write_array_to_binary(subfiles.decompress.out_xdata, xdata, ap->len);
    }
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
