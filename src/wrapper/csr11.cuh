/**
 * @file csr11.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-28
 * (rev) 2022-01-10
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_CSR11_CUH
#define CUSZ_CSR11_CUH

// caveat: CUDA 10.2 may fail.

#include <clocale>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <driver_types.h>

#include "../../include/reducer.hh"
#include "../common.hh"
#include "../utils.hh"

// clang-format off
template <typename F> struct cuszCUSPARSE;
template <> struct cuszCUSPARSE<float>  { const static cudaDataType type = CUDA_R_32F; };
template <> struct cuszCUSPARSE<double> { const static cudaDataType type = CUDA_R_64F; };
// clang-format on

/******************************************************************************
                            macros for shorthand writing
 ******************************************************************************/

#define CSR11_FREEDEV(VAR)             \
    if (d_##VAR) {                     \
        CHECK_CUDA(cudaFree(d_##VAR)); \
        d_##VAR = nullptr;             \
    }

#define DEVICE2DEVICE_COPY(VAR, FIELD)                                                                 \
    {                                                                                                  \
        auto dst = d_csr + header.entry[HEADER::FIELD];                                                \
        auto src = reinterpret_cast<BYTE*>(d_##VAR);                                                   \
        CHECK_CUDA(cudaMemcpyAsync(dst, src, nbyte[HEADER::FIELD], cudaMemcpyDeviceToDevice, stream)); \
    }

#define CSR11_ALLOCDEV(VAR, SYM)                           \
    CHECK_CUDA(cudaMalloc(&d_##VAR, rte.nbyte[RTE::SYM])); \
    CHECK_CUDA(cudaMemset(d_##VAR, 0x0, rte.nbyte[RTE::SYM]));

#define DEFINE_CSR11_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};

/******************************************************************************
                               class definition
******************************************************************************/

namespace cusz {

template <typename T = float>
class CSR11 : public VirtualGatherScatter {
   public:
    using Origin                  = T;
    static const auto DEFAULT_LOC = cusz::LOC::DEVICE;

    using BYTE      = uint8_t;
    using MetadataT = uint32_t;

   public:
    /******************************************************************************
                                   header definition
     ******************************************************************************/
    struct header_t {
        static const int HEADER = 0;
        static const int ROWPTR = 1;
        static const int COLIDX = 2;
        static const int VAL    = 3;
        static const int END    = 4;

        int    header_nbyte : 16;
        size_t uncompressed_len;  // TODO unnecessary?
        int    m;                 // as well as n; square

        // TODO compatibility checking
#if CUDART_VERSION >= 11020
// TODO
#elif CUDART_VERSION >= 10000
// TODO
#else
// throw
#endif
        int64_t   nnz;
        MetadataT entry[END + 1];

        MetadataT subfile_size() const { return entry[END]; }
    };
    using HEADER = struct header_t;

    /******************************************************************************
                                     runtime helper
     ******************************************************************************/
    struct runtime_encode_helper {
        static const int CSR    = 0;
        static const int ROWPTR = 1;
        static const int COLIDX = 2;
        static const int VAL    = 3;
        static const int END    = 4;

        uint32_t nbyte[END];

#if CUDART_VERSION >= 11020
        cusparseHandle_t     handle{nullptr};
        cusparseSpMatDescr_t spmat;
        cusparseDnMatDescr_t dnmat;

        void*  d_buffer{nullptr};
        size_t d_buffer_size{0};
#elif CUDART_VERSION >= 10000
        cusparseHandle_t   handle{nullptr};
        cusparseMatDescr_t mat_desc{nullptr};

        size_t lwork_in_bytes{0};
        char*  d_work{nullptr};
#endif
        uint32_t m{0};
        int64_t  nnz{0};
        HEADER*  ptr_header{nullptr};
    };
    using RTE = runtime_encode_helper;
    RTE rte;

   private:
    /******************************************************************************
                                  class-side variables
     ******************************************************************************/
    DEFINE_CSR11_ARRAY(csr, BYTE);
    DEFINE_CSR11_ARRAY(rowptr, int);
    DEFINE_CSR11_ARRAY(colidx, int);
    DEFINE_CSR11_ARRAY(val, T);

   private:
    float milliseconds{0.0};

    Capsule<int> rowptr;
    Capsule<int> colidx;
    Capsule<T>   values;

    /**
     * @brief Internal use when the real nnz is known.
     *
     * @param nnz
     */
    void reconfigure_with_precise_nnz(int nnz)
    {
        // TODO
    }

#if CUDART_VERSION >= 11020

    /**
     * @brief Internal gather method; gather method as of CUDA 11.2
     *
     * @param in_dense (device array) input "as" dense-format m-by-m matrix
     * @param stream CUDA stream
     */
    void gather_CUDA11_new(T* in_dense, cudaStream_t stream)
    {
        auto num_rows = rte.m;
        auto num_cols = rte.m;
        auto ld       = rte.m;

        auto gather11_init_mat = [&]() {
            // create dense matrix wrapper
            CHECK_CUSPARSE(cusparseCreateDnMat(
                &rte.dnmat, num_rows, num_cols, ld, in_dense, cuszCUSPARSE<T>::type, CUSPARSE_ORDER_ROW));

            // create CSR wrapper
            CHECK_CUSPARSE(cusparseCreateCsr(
                &rte.spmat, num_rows, num_cols, 0, d_rowptr, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, cuszCUSPARSE<T>::type));
        };

        auto gather11_init_buffer = [&](){
        {  // allocate an external buffer if needed
            cuda_timer_t t;
            t.timer_start(stream);

            CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
                rte.handle, rte.dnmat, rte.spmat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &rte.d_buffer_size));

            t.timer_end(stream);
            milliseconds += t.get_time_elapsed();

            CHECK_CUDA(cudaMalloc(&rte.d_buffer, rte.d_buffer_size));
        }};

        auto gather11_analysis = [&]() {
            cuda_timer_t t;
            t.timer_start(stream);

            CHECK_CUSPARSE(cusparseDenseToSparse_analysis(
                rte.handle, rte.dnmat, rte.spmat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, rte.d_buffer));

            t.timer_end(stream);
            milliseconds += t.get_time_elapsed();
        };

        int64_t num_rows_tmp, num_cols_tmp;

        auto gather11_get_nnz = [&]() {
            // get number of non-zero elements
            CHECK_CUSPARSE(cusparseSpMatGetSize(rte.spmat, &num_rows_tmp, &num_cols_tmp, &rte.nnz));
        };

        auto gather11_get_rowptr = [&]() {
            // reset offsets, column indices, and values pointers
            CHECK_CUSPARSE(cusparseCsrSetPointers(rte.spmat, d_rowptr, d_colidx, d_val));
        };

        auto gather11_dn2csr = [&]() {
            cuda_timer_t t;
            t.timer_start(stream);

            CHECK_CUSPARSE(cusparseDenseToSparse_convert(
                rte.handle, rte.dnmat, rte.spmat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, rte.d_buffer));

            t.timer_end(stream);
            milliseconds += t.get_time_elapsed();
        };

        /********************************************************************************/
        milliseconds = 0;

        CHECK_CUSPARSE(cusparseCreate(&rte.handle));
        if (stream) CHECK_CUSPARSE(cusparseSetStream(rte.handle, stream));  // TODO move out

        gather11_init_mat();
        gather11_init_buffer();
        gather11_analysis();
        gather11_get_nnz();
        gather11_get_rowptr();
        gather11_dn2csr();

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE(cusparseDestroyDnMat(rte.dnmat));
        CHECK_CUSPARSE(cusparseDestroySpMat(rte.spmat));
        CHECK_CUSPARSE(cusparseDestroy(rte.handle));
    }

#elif CUDART_VERSION >= 10000

    /**
     * @brief Internal gather method; CUDA version >= 10.0 * < 11.2
     *
     * @param in_dense (device array) input "as" dense-format m-by-m matrix
     * @param stream CUDA stream
     */
    void gather_CUDA10_new(T* in_dense, cudaStream_t stream = nullptr)
    {
        int num_rows, num_cols, ld;
        num_rows = num_cols = ld = rte.m;

        float threshold{0};
        auto has_ext_stream{false};

        /******************************************************************************/

        auto gather10_init_and_probe = [&]() {
            {  // init

                CHECK_CUSPARSE(cusparseCreateMatDescr(&rte.mat_desc));  // 4. create rte.mat_desc
                CHECK_CUSPARSE(cusparseSetMatIndexBase(rte.mat_desc, CUSPARSE_INDEX_BASE_ZERO));  // zero based
                CHECK_CUSPARSE(cusparseSetMatType(rte.mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL));   // type
            }

            {  // probe
                cuda_timer_t t;
                t.timer_start(stream);

                CHECK_CUSPARSE(cusparseSpruneDense2csr_bufferSizeExt(
                    rte.handle, num_rows, num_cols, in_dense, ld, &threshold, rte.mat_desc, d_val, d_rowptr, d_colidx,
                    &rte.lwork_in_bytes));

                t.timer_end(stream);
                milliseconds += t.get_time_elapsed();
            }

            if (nullptr != rte.d_work) cudaFree(rte.d_work);
            CHECK_CUDA(cudaMalloc((void**)&rte.d_work, rte.lwork_in_bytes));  // TODO where to release d_work?
        };

        auto gather10_compute_rowptr_and_nnz = [&]() {  // step 4
            cuda_timer_t t;
            t.timer_start(stream);

            int nnz;  // for compatibility; cuSPARSE of CUDA 11 changed data type

            CHECK_CUSPARSE(cusparseSpruneDense2csrNnz(
                rte.handle, num_rows, num_cols, in_dense, ld, &threshold, rte.mat_desc, d_rowptr, &nnz, rte.d_work));

            t.timer_end(stream);
            milliseconds += t.get_time_elapsed();
            CHECK_CUDA(cudaStreamSynchronize(stream));

            rte.nnz = nnz;
        };

        auto gather10_compute_colidx_and_val = [&]() {  // step 5
            cuda_timer_t t;
            t.timer_start(stream);

            CHECK_CUSPARSE(cusparseSpruneDense2csr(  //
                rte.handle, num_rows, num_cols, in_dense, ld, &threshold, rte.mat_desc, d_val, d_rowptr, d_colidx,
                rte.d_work));

            t.timer_end(stream);
            milliseconds += t.get_time_elapsed();
            CHECK_CUDA(cudaStreamSynchronize(stream));
        };

        /********************************************************************************/
        milliseconds = 0;

        if (stream)
            has_ext_stream = true;
        else
            CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));  // 1. create stream
        CHECK_CUSPARSE(cusparseCreate(&rte.handle));                                // 2. create handle
        CHECK_CUSPARSE(cusparseSetStream(rte.handle, stream));                      // 3. bind stream

        gather10_init_and_probe();
        gather10_compute_rowptr_and_nnz();
        if (rte.nnz == 0) { return; }
        gather10_compute_colidx_and_val();

        // TODO no need to destroy?
        if (rte.handle) cusparseDestroy(rte.handle);
        if (rte.mat_desc) cusparseDestroyMatDescr(rte.mat_desc);
        if ((not has_ext_stream) and stream) cudaStreamDestroy(stream);
        /********************************************************************************/
    }

#endif

#if CUDART_VERSION >= 11020

    /**
     * @brief Internal scatter method; use as of CUDA 11.2
     *
     * @param in_csr (device array) input CSR-format
     * @param out_dense (device array) output dense-format
     * @param stream CUDA stream
     * @param header_on_device (optional) configuration; if true, the header is copied from the on-device binary.
     */
    void scatter_CUDA11_new(BYTE* in_csr, T* out_dense, cudaStream_t stream = nullptr, bool header_on_device = true)
    {
        header_t header;
        if (header_on_device)
            CHECK_CUDA(cudaMemcpyAsync(&header, in_csr, sizeof(header), cudaMemcpyDeviceToHost, stream));

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_csr + header.entry[HEADER::SYM])
        auto d_rowptr = ACCESSOR(ROWPTR, int);
        auto d_colidx = ACCESSOR(COLIDX, int);
        auto d_val    = ACCESSOR(VAL, T);
#undef ACCESSOR

        auto num_rows = header.m;
        auto num_cols = header.m;
        auto ld       = header.m;
        auto nnz      = header.nnz;

        auto scatter11_init_mat = [&]() {
            CHECK_CUSPARSE(cusparseCreateCsr(
                &rte.spmat, num_rows, num_cols, nnz, d_rowptr, d_colidx, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, cuszCUSPARSE<T>::type));

            CHECK_CUSPARSE(cusparseCreateDnMat(
                &rte.dnmat, num_rows, num_cols, ld, out_dense, cuszCUSPARSE<T>::type, CUSPARSE_ORDER_ROW));
        };

        auto scatter11_init_buffer = [&]() {
            cuda_timer_t t;
            t.timer_start(stream);

            // allocate an external buffer if needed
            CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(
                rte.handle, rte.spmat, rte.dnmat, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &rte.d_buffer_size));

            t.timer_end(stream);
            milliseconds += t.get_time_elapsed();

            CHECK_CUDA(cudaMalloc(&rte.d_buffer, rte.d_buffer_size));
        };

        auto scatter11_csr2dn = [&]() {
            cuda_timer_t t;
            t.timer_start(stream);

            CHECK_CUSPARSE(cusparseSparseToDense(
                rte.handle, rte.spmat, rte.dnmat, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, rte.d_buffer));

            t.timer_end(stream);
            milliseconds += t.get_time_elapsed();
        };

        /******************************************************************************/
        milliseconds = 0;

        CHECK_CUSPARSE(cusparseCreate(&rte.handle));
        if (stream) CHECK_CUSPARSE(cusparseSetStream(rte.handle, stream));

        scatter11_init_mat();
        scatter11_init_buffer();
        scatter11_csr2dn();

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE(cusparseDestroySpMat(rte.spmat));
        CHECK_CUSPARSE(cusparseDestroyDnMat(rte.dnmat));
        CHECK_CUSPARSE(cusparseDestroy(rte.handle));
    }

#elif CUDART_VERSION >= 10000

    /**
     * @brief Internal scatter method; CUDA version >= 10.0 * < 11.2
     *
     * @param in_csr (device array) input CSR-format
     * @param out_dense (device array) output dense-format
     * @param stream CUDA stream
     * @param header_on_device (optional) configuration; if true, the header is copied from the on-device binary.
     */
    void scatter_CUDA10_new(BYTE* in_csr, T* out_dense, cudaStream_t stream = nullptr, bool header_on_device = true)
    {
        header_t header;
        if (header_on_device)
            CHECK_CUDA(cudaMemcpyAsync(&header, in_csr, sizeof(header), cudaMemcpyDeviceToHost, stream));

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_csr + header.entry[HEADER::SYM])
        auto d_rowptr = ACCESSOR(ROWPTR, int);
        auto d_colidx = ACCESSOR(COLIDX, int);
        auto d_val = ACCESSOR(VAL, T);
#undef ACCESSOR

        auto num_rows = header.m;
        auto num_cols = header.m;
        auto ld = header.m;

        auto has_external_stream = false;

        /******************************************************************************/

        auto scatter10_init = [&]() {
            CHECK_CUSPARSE(cusparseCreateMatDescr(&rte.mat_desc));                            // 4. create descr
            CHECK_CUSPARSE(cusparseSetMatIndexBase(rte.mat_desc, CUSPARSE_INDEX_BASE_ZERO));  // zero based
            CHECK_CUSPARSE(cusparseSetMatType(rte.mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL));   // type
        };

        auto scatter10_sparse2dense = [&]() {
            cuda_timer_t t;
            t.timer_start(stream);

            CHECK_CUSPARSE(cusparseScsr2dense(
                rte.handle, num_rows, num_cols, rte.mat_desc, d_val, d_rowptr, d_colidx, out_dense, ld));

            t.timer_end();
            milliseconds += t.get_time_elapsed();
            CHECK_CUDA(cudaStreamSynchronize(stream));
        };

        /******************************************************************************/
        if (stream)
            has_external_stream = true;
        else
            CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK_CUSPARSE(cusparseCreate(&rte.handle));
        CHECK_CUSPARSE(cusparseSetStream(rte.handle, stream));

        scatter10_init();
        scatter10_sparse2dense();

        if (rte.handle) cusparseDestroy(rte.handle);
        if (rte.mat_desc) cusparseDestroyMatDescr(rte.mat_desc);
        if ((not has_external_stream) and stream) cudaStreamDestroy(stream);
        /******************************************************************************/
    }

#endif

   public:
    float get_time_elapsed() const { return milliseconds; }

    CSR11() = default;

    ~CSR11()
    {
        CSR11_FREEDEV(csr);
        CSR11_FREEDEV(rowptr);
        CSR11_FREEDEV(colidx);
        CSR11_FREEDEV(val);
    }

    // only placeholding
    void scatter() {}
    void gather() {}

    /**
     * @brief Allocate according to the input; usually allocate much more than the actual need at runtime for failsafe.
     *
     * @param in_uncompressed_len (host variable) input length
     * @param dbg_print print for debugging
     */
    void allocate_workspace(size_t const in_uncompressed_len, bool dbg_print = false)
    {
        auto max_compressed_bytes = [&]() { return in_uncompressed_len / 10 * sizeof(T); };
        auto init_nnz             = [&]() { return in_uncompressed_len / 10; };
        auto debug                = [&]() {
            setlocale(LC_NUMERIC, "");

#define PRINT_DBG(VAR) printf("nbyte-%-*s:  %'10u\n", 10, #VAR, rte.nbyte[RTE::VAR]);
            printf("\nCSR11::allocate_workspace() debugging:\n");
            printf("%-*s:  %'10ld\n", 16, "init.nnz", init_nnz());
            PRINT_DBG(CSR);
            PRINT_DBG(ROWPTR);
            PRINT_DBG(COLIDX);
            PRINT_DBG(VAL);
            printf("\n");
#undef PRINT_DBG
        };

        memset(rte.nbyte, 0, sizeof(uint32_t) * RTE::END);

        rte.m   = Reinterpret1DTo2D::get_square_size(in_uncompressed_len);
        rte.nnz = init_nnz();

        rte.nbyte[RTE::CSR]    = max_compressed_bytes();
        rte.nbyte[RTE::ROWPTR] = sizeof(int) * (rte.m + 1);
        rte.nbyte[RTE::COLIDX] = sizeof(int) * init_nnz();
        rte.nbyte[RTE::VAL]    = sizeof(T) * init_nnz();

        CSR11_ALLOCDEV(csr, CSR);
        CSR11_ALLOCDEV(rowptr, ROWPTR);
        CSR11_ALLOCDEV(colidx, COLIDX);
        CSR11_ALLOCDEV(val, VAL);

        if (dbg_print) debug();
    }

   private:
    /**
     * @brief Collect fragmented arrays.
     *
     * @param header (host variable)
     * @param in_uncompressed_len (host variable)
     * @param stream CUDA stream
     */
    void
    subfile_collect(HEADER& header, size_t in_uncompressed_len, cudaStream_t stream = nullptr, bool dbg_print = false)
    {
        header.header_nbyte     = sizeof(HEADER);
        header.uncompressed_len = in_uncompressed_len;
        header.nnz              = rte.nnz;
        header.m                = rte.m;

        // update (redundant here)
        rte.nbyte[RTE::COLIDX] = sizeof(int) * rte.nnz;
        rte.nbyte[RTE::VAL]    = sizeof(T) * rte.nnz;

        MetadataT nbyte[HEADER::END];
        nbyte[HEADER::HEADER] = 128;
        nbyte[HEADER::ROWPTR] = rte.nbyte[RTE::ROWPTR];
        nbyte[HEADER::COLIDX] = rte.nbyte[RTE::COLIDX];
        nbyte[HEADER::VAL]    = rte.nbyte[RTE::VAL];

        header.entry[0] = 0;
        // *.END + 1; need to knwo the ending position
        for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] = nbyte[i - 1]; }
        for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] += header.entry[i - 1]; }

        auto debug_header_entry = [&]() {
            printf("\nCSR11::subfile_collect() debugging:\n");
            printf("%-*s:  %'10ld\n", 16, "final.nnz", rte.nnz);
            printf("  ENTRIES\n");

#define PRINT_ENTRY(VAR) printf("%d %-*s:  %'10u\n", (int)HEADER::VAR, 14, #VAR, header.entry[HEADER::VAR]);
            PRINT_ENTRY(HEADER);
            PRINT_ENTRY(ROWPTR);
            PRINT_ENTRY(COLIDX);
            PRINT_ENTRY(VAL);
            PRINT_ENTRY(END);
            printf("\n");
#undef PRINT_ENTRY
        };
        if (dbg_print) debug_header_entry();

        CHECK_CUDA(cudaMemcpyAsync(d_csr, &header, sizeof(header), cudaMemcpyHostToDevice, stream));

        /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

        DEVICE2DEVICE_COPY(rowptr, ROWPTR)
        DEVICE2DEVICE_COPY(colidx, COLIDX)
        DEVICE2DEVICE_COPY(val, VAL)

        /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));
    }

   public:
    /**
     * @brief
     *
     */
    void clear_buffer()
    {
        cudaMemset(d_csr, 0x0, rte.nbyte[RTE::CSR]);
        cudaMemset(d_rowptr, 0x0, rte.nbyte[RTE::ROWPTR]);
        cudaMemset(d_colidx, 0x0, rte.nbyte[RTE::COLIDX]);
        cudaMemset(d_val, 0x0, rte.nbyte[RTE::VAL]);
    }

    /**
     * @brief Public interface for gather method.
     *
     * @param in_uncompressed (device array) input
     * @param in_uncompressed_len (host variable) input length
     * @param out_compressed (device array) reference output
     * @param out_compressed_len (host variable) reference output length
     * @param stream CUDA stream
     */
    void gather(
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        BYTE*&       out_compressed,
        size_t&      out_compressed_len,
        cudaStream_t stream    = nullptr,
        bool         dbg_print = false)
    {
        // cautious!
        HEADER header;
        rte.ptr_header = &header;

#if CUDART_VERSION >= 11020
        gather_CUDA11_new(in_uncompressed, stream);
#elif CUDART_VERSION >= 10000
        gather_CUDA10_new(in_uncompressed, stream);
#endif

        subfile_collect(header, in_uncompressed_len, stream, dbg_print);

        out_compressed     = d_csr;
        out_compressed_len = header.subfile_size();
    }

    /**
     * @brief Public interface for scatter.
     *
     * @param in_compressed (device array)
     * @param out_decompressed (device array)
     * @param stream CUDA stream
     * @param header_on_device (optional) configuration; if true, the header is copied from the on-device binary.
     */
    void scatter(
        BYTE*        in_compressed,  //
        T*           out_decompressed,
        cudaStream_t stream           = nullptr,
        bool         header_on_device = true)
    {
        header_t header;
        if (header_on_device)
            CHECK_CUDA(cudaMemcpyAsync(&header, in_compressed, sizeof(header), cudaMemcpyDeviceToHost, stream));

#if CUDART_VERSION >= 11020
        scatter_CUDA11_new(in_compressed, out_decompressed, stream);
#elif CUDART_VERSION >= 10000
        scatter_CUDA10_new(in_compressed, out_decompressed, stream);
#endif
    }

    // end of class
};

//
}  // namespace cusz

#undef DEVICE2DEVICE_COPY
#undef CSR11_ALLCDEV
#undef CSR11_FREEDEV
#undef DEFINE_CSR11_ARRAY

#endif