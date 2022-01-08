/**
 * @file csr11.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-28
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_HANDLE_SPARSITY11_CUH
#define CUSZ_WRAPPER_HANDLE_SPARSITY11_CUH

// caveat: CUDA 11.2 starts introduce new cuSAPARSE API, which cannot be used prior to 11.2.

#include <driver_types.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#if CUDART_VERSION >= 11020
#include <cusparse.h>
#elif CUDART_VERSION >= 10000
#endif

#include "../../include/reducer.hh"

// clang-format off
template <typename F> struct cuszCUSPARSE;
template <> struct cuszCUSPARSE<float>  { const static cudaDataType type = CUDA_R_32F; };
template <> struct cuszCUSPARSE<double> { const static cudaDataType type = CUDA_R_64F; };
// clang-format on

namespace cusz {

template <typename T = float>
class CSR11 : public VirtualGatherScatter {
   public:
    using Origin                  = T;
    static const auto DEFAULT_LOC = cusz::LOC::DEVICE;

    using BYTE      = uint8_t;
    using MetadataT = uint32_t;

   public:
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
        uint32_t m;
        int64_t  nnz;
        HEADER*  ptr_header;
    };
    using RTE = runtime_encode_helper;
    RTE rte;

   private:
#define DEFINE_CSR11_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};
    DEFINE_CSR11_ARRAY(csr, BYTE);
    DEFINE_CSR11_ARRAY(rowptr, int);
    DEFINE_CSR11_ARRAY(colidx, int);
    DEFINE_CSR11_ARRAY(val, T);
#undef DEFINE_CSR11_ARRAY

   private:
    // clang-format off
    uint8_t* pool_ptr;
    struct { unsigned int rowptr, colidx, values; } offset;
    struct { unsigned int rowptr, colidx, values, total; } nbyte;
    unsigned int workspace_nbyte, dump_nbyte;
    unsigned int m{0}, dummy_nnz{0}, nnz{0};
    float milliseconds{0.0};
    // clang-format on

    Capsule<int> rowptr;
    Capsule<int> colidx;
    Capsule<T>   values;

    // use when the real nnz is known
    void reconfigure_with_precise_nnz(int nnz);

#if CUDART_VERSION >= 11020
    void gather_CUDA11(T* in_dense, unsigned int& dump_nbyte, cudaStream_t = nullptr);
    void gather_CUDA11_new(T* in_dense, cudaStream_t = nullptr);
#elif CUDART_VERSION >= 10000
    void gather_CUDA10(T* in_dense, unsigned int& dump_nbyte, cudaStream_t = nullptr);
    void gather_CUDA10_new(T* in_dense, cudaStream_t = nullptr);
#else
#error CUDART_VERSION must be no less than 10.0!
#endif

#if CUDART_VERSION >= 11020
    void scatter_CUDA11(T* out_dense, cudaStream_t stream = nullptr);
    void scatter_CUDA11_new(BYTE* in_csr, T* out_dense, cudaStream_t = nullptr, bool header_on_device = true);
#elif CUDART_VERSION >= 10000
    void scatter_CUDA10(T* out_dense, cudaStream_t stream = nullptr);
    void scatter_CUDA10_new(BYTE* in_csr, T* out_dense, cudaStream_t = nullptr, bool header_on_device = true);
#else
#error CUDART_VERSION must be no less than 10.0!
#endif

    // TODO handle nnz == 0 otherwise
    unsigned int query_csr_bytelen() const
    {
        return sizeof(int) * (m + 1)  // rowptr
               + sizeof(int) * nnz    // colidx
               + sizeof(T) * nnz;     // values
    }

    void extract(uint8_t* _pool);

   public:
    // helper

    static uint32_t get_total_nbyte(uint32_t len, int nnz)
    {
        auto m = Reinterpret1DTo2D::get_square_size(len);
        return sizeof(int) * (m + 1) + sizeof(int) * nnz + sizeof(T) * nnz;
    }

    float get_time_elapsed() const { return milliseconds; }

    CSR11() = default;

    ~CSR11()
    {
#define CSR11_FREEDEV(VAR, SYM) \
    if (d_##VAR) {              \
        cudaFree(d_##VAR);      \
        d_##VAR = nullptr;      \
    }

        CSR11_FREEDEV(csr, CSR);
        CSR11_FREEDEV(rowptr, ROWPTR);
        CSR11_FREEDEV(colidx, COLIDX);
        CSR11_FREEDEV(val, VAL);

#undef CSR11_FREEDEV
    }

    template <cusz::LOC FROM = cusz::LOC::DEVICE, cusz::LOC TO = cusz::LOC::HOST>
    CSR11& consolidate(uint8_t* dst);  //, cudaMemcpyKind direction = cudaMemcpyDeviceToHost);

    CSR11& decompress_set_nnz(unsigned int _nnz);

    void gather(
        T*           in,
        uint32_t     in_len,
        int*         out_rowptr,
        int*&        out_colidx,
        T*&          out_val,
        int&         out_nnz,
        uint32_t&    nbyte_dump,
        cudaStream_t stream = nullptr)
    {
        m = Reinterpret1DTo2D::get_square_size(in_len);

        if (out_rowptr) rowptr.template shallow_copy<DEFAULT_LOC>(out_rowptr);
        colidx.template shallow_copy<DEFAULT_LOC>(out_colidx);
        values.template shallow_copy<DEFAULT_LOC>(out_val);

#if CUDART_VERSION >= 11020
        gather_CUDA11(in, nbyte_dump, stream);
#elif CUDART_VERSION >= 10000
        gather_CUDA10(in, nbyte_dump, stream);
#else
#error CUDART_VERSION must be no less than 10.0!
#endif
        out_nnz = this->nnz;
    }

    // only placeholding
    void scatter() {}
    void gather() {}

    void scatter(uint8_t* _pool, int nnz, T* out, uint32_t out_len, cudaStream_t stream = nullptr)
    {
        m = Reinterpret1DTo2D::get_square_size(out_len);
        decompress_set_nnz(nnz);
        extract(_pool);

#if CUDART_VERSION >= 11020
        scatter_CUDA11(out, stream);
#elif CUDART_VERSION >= 10000
        scatter_CUDA10(out, stream);
#else
#error CUDART_VERSION must be no less than 10.0!
#endif
    }

    ////////////////////////////////////////////////////////////////////////////////
    static void get_padding(size_t const in_uncompressed_len, size_t& padded_len, int& m)
    {
        m          = Reinterpret1DTo2D::get_square_size(in_uncompressed_len);
        padded_len = m * m;
    }

    void allocate_workspace(size_t const in_uncompressed_len)
    {
        auto max_compressed_bytes = [&]() { return in_uncompressed_len / 10 * sizeof(T); };
        auto init_nnz             = [&]() { return in_uncompressed_len / 10; };

        memset(rte.nbyte, 0, sizeof(uint32_t) * RTE::END);

        rte.m = Reinterpret1DTo2D::get_square_size(in_uncompressed_len);

        rte.nbyte[RTE::CSR]    = max_compressed_bytes();
        rte.nbyte[RTE::ROWPTR] = sizeof(int) * (rte.m + 1);
        rte.nbyte[RTE::COLIDX] = sizeof(int) * init_nnz();
        rte.nbyte[RTE::VAL]    = sizeof(T) * init_nnz();

#define CSR11_ALLOCDEV(VAR, SYM)               \
    cudaMalloc(&d_##VAR, rte.nbyte[RTE::SYM]); \
    cudaMemset(d_##VAR, 0x0, rte.nbyte[RTE::SYM]);
        CSR11_ALLOCDEV(csr, CSR);
        CSR11_ALLOCDEV(rowptr, ROWPTR);
        CSR11_ALLOCDEV(colidx, COLIDX);
        CSR11_ALLOCDEV(val, VAL);
#undef CSR11_ALLCDEV
    }

    void gather_new(
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        BYTE*&       out_compressed,
        size_t&      out_compressed_len,
        cudaStream_t stream = nullptr)
    {
        // cautious!
        HEADER header;
        rte.ptr_header = &header;

        auto subfile_collect = [&]() {
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
                for (auto i = 0; i < HEADER::END + 1; i++) printf("%d, header entry: %d\n", i, header.entry[i]);
            };
            debug_header_entry();

            CHECK_CUDA(cudaMemcpyAsync(d_csr, &header, sizeof(header), cudaMemcpyHostToDevice, stream));

            /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

#define DEVICE2DEVICE_COPY(VAR, FIELD)                                                                 \
    {                                                                                                  \
        auto dst = d_csr + header.entry[HEADER::FIELD];                                                \
        auto src = reinterpret_cast<BYTE*>(d_##VAR);                                                   \
        CHECK_CUDA(cudaMemcpyAsync(dst, src, nbyte[HEADER::FIELD], cudaMemcpyDeviceToDevice, stream)); \
    }
            DEVICE2DEVICE_COPY(rowptr, ROWPTR)
            DEVICE2DEVICE_COPY(colidx, COLIDX)
            DEVICE2DEVICE_COPY(val, VAL)
#undef DEVICE2DEVICE_COPY

            /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));
        };

        // -----------------------------------------------------------------------------
#if CUDART_VERSION >= 11020
        gather_CUDA11_new(in_uncompressed, stream);
#elif CUDART_VERSION >= 10000
        gather_CUDA10_new(in_uncompressed, stream);
#else
#error CUDART_VERSION must be no less than 10.0!
#endif

        subfile_collect();

        out_compressed     = d_csr;
        out_compressed_len = header.subfile_size();
    }

    void scatter_new(
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
#else
#error CUDART_VERSION must be no less than 10.0!
#endif
    }
};

//
}  // namespace cusz

#endif