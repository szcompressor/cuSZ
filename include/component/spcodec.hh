/**
 * @file spcodec.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_COMPONENT_SPCODECS_HH
#define CUSZ_COMPONENT_SPCODECS_HH

#define DEFINE_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdint>
#include <memory>

namespace cusz {

template <typename T, typename M = uint32_t>
class SpcodecCSR {
   public:
    using Origin = T;
    using BYTE   = uint8_t;

   private:
    class impl;
    std::unique_ptr<impl> pimpl;

   public:
    ~SpcodecCSR();                             // dtor
    SpcodecCSR();                              // ctor
    SpcodecCSR(const SpcodecCSR&);             // copy ctor
    SpcodecCSR& operator=(const SpcodecCSR&);  // copy assign
    SpcodecCSR(SpcodecCSR&&);                  // move ctor
    SpcodecCSR& operator=(SpcodecCSR&&);       // move assign

    void init(size_t const, int = 4, bool = false);
    void encode(T*, size_t const, BYTE*&, size_t&, cudaStream_t = nullptr, bool = false);
    void decode(BYTE*, T*, cudaStream_t = nullptr);
    void clear_buffer();
    // getter
    float get_time_elapsed() const;
};

template <typename T, typename M>
class SpcodecCSR<T, M>::impl {
   public:
    using Origin = T;
    using BYTE   = uint8_t;
    // using MetadataT = uint32_t;
    using MetadataT = M;

    struct Header;
    struct runtime_encode_helper;
    using RTE = struct runtime_encode_helper;

   private:
    // static const auto DEFAULT_LOC = cusz::LOC::DEVICE;

    RTE   rte;
    float milliseconds{0.0};

    DEFINE_ARRAY(csr, BYTE);
    DEFINE_ARRAY(rowptr, int);
    DEFINE_ARRAY(colidx, int);
    DEFINE_ARRAY(val, T);

   private:
    void gather_CUDA_11020(T*, cudaStream_t = nullptr);
    void scatter_CUDA_11020(BYTE*, T*, cudaStream_t = nullptr, bool = true);
    void gather_CUDA_fallback(T*, cudaStream_t = nullptr);
    void scatter_CUDA_fallback(BYTE*, T*, cudaStream_t = nullptr, bool = true);
    void subfile_collect(Header&, size_t, cudaStream_t = nullptr, bool = false);

   public:
    impl() = default;
    ~impl();
    void init(size_t const, int = 4, bool = false);
    void encode(T*, size_t const, BYTE*&, size_t&, cudaStream_t = nullptr, bool = false);
    void decode(BYTE*, T*, cudaStream_t = nullptr);
    void clear_buffer();
    // getter
    float get_time_elapsed() const;
};

template <typename T, typename M>
struct SpcodecCSR<T, M>::impl::Header {
    static const int HEADER = 0;
    static const int ROWPTR = 1;
    static const int COLIDX = 2;
    static const int VAL    = 3;
    static const int END    = 4;

    int     header_nbyte : 16;
    size_t  uncompressed_len;  // TODO unnecessary?
    int     m;                 // as well as n; square
    int64_t nnz;

    MetadataT entry[END + 1];

    MetadataT subfile_size() const { return entry[END]; }
};

template <typename T, typename M>
struct SpcodecCSR<T, M>::impl::runtime_encode_helper {
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
    Header*  ptr_header{nullptr};
};

}  // namespace cusz

#undef DEFINE_ARRAY

#endif
