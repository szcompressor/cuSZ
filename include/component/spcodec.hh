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

#include <hip/hip_runtime.h>
#include <hipsparse.h>

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
    void encode(T*, size_t const, BYTE*&, size_t&, hipStream_t = nullptr, bool = false);
    void decode(BYTE*, T*, hipStream_t = nullptr);
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
    void gather_CUDA_11020(T*, hipStream_t = nullptr);
    void scatter_CUDA_11020(BYTE*, T*, hipStream_t = nullptr, bool = true);
    void gather_CUDA_fallback(T*, hipStream_t = nullptr);
    void scatter_CUDA_fallback(BYTE*, T*, hipStream_t = nullptr, bool = true);
    void subfile_collect(Header&, size_t, hipStream_t = nullptr, bool = false);

   public:
    impl() = default;
    ~impl();
    void init(size_t const, int = 4, bool = false);
    void encode(T*, size_t const, BYTE*&, size_t&, hipStream_t = nullptr, bool = false);
    void decode(BYTE*, T*, hipStream_t = nullptr);
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

    hipsparseHandle_t   handle{nullptr};
    hipsparseMatDescr_t mat_desc{nullptr};

    size_t lwork_in_bytes{0};
    char*  d_work{nullptr};
    uint32_t m{0};
    int64_t  nnz{0};
    Header*  ptr_header{nullptr};
};

/*******************************************************************************
 * sparsity-aware coder/decoder, vector
 *******************************************************************************/

template <typename T, typename M = uint32_t>
class SpcodecVec {
   public:
    using Origin = T;
    using BYTE   = uint8_t;

   private:
    class impl;
    std::unique_ptr<impl> pimpl;

   public:
    ~SpcodecVec();                             // dtor
    SpcodecVec();                              // ctor
    SpcodecVec(const SpcodecVec&);             // copy ctor
    SpcodecVec& operator=(const SpcodecVec&);  // copy assign
    SpcodecVec(SpcodecVec&&);                  // move ctor
    SpcodecVec& operator=(SpcodecVec&&);       // move assign

    void init(size_t const, int = 4, bool = false);
    void encode(T*, size_t const, BYTE*&, size_t&, hipStream_t = nullptr, bool = false);
    void decode(BYTE*, T*, hipStream_t = nullptr);
    void clear_buffer();
    // getter
    float get_time_elapsed() const;
};

template <typename T, typename M>
struct SpcodecVec<T, M>::impl {
   public:
    using Origin    = T;
    using BYTE      = uint8_t;
    using MetadataT = M;

   private:
    DEFINE_ARRAY(spfmt, BYTE);
    DEFINE_ARRAY(idx, M);
    DEFINE_ARRAY(val, T);

    struct Header;
    struct runtime_encode_helper;
    using header_t = Header;
    using RTE      = runtime_encode_helper;

    float milliseconds{0.0};

    RTE rte;

   private:
    void subfile_collect(Header&, size_t, hipStream_t = nullptr, bool = false);

   public:
    impl() = default;
    ~impl();
    void init(size_t const, int = 4, bool = false);
    void encode(T*, size_t const, BYTE*&, size_t&, hipStream_t = nullptr, bool = false);
    void decode(BYTE*, T*, hipStream_t = nullptr);
    void clear_buffer();
    // getter
    float get_time_elapsed() const;
};

template <typename T, typename M>
struct SpcodecVec<T, M>::impl::Header {
    static const int HEADER = 0;
    static const int IDX    = 1;
    static const int VAL    = 2;
    static const int END    = 3;

    int       header_nbyte : 16;
    size_t    uncompressed_len;
    int       nnz;
    MetadataT entry[END + 1];

    MetadataT subfile_size() const { return entry[END]; }
};

template <typename T, typename M>
struct SpcodecVec<T, M>::impl::runtime_encode_helper {
    static const int SPFMT = 0;
    static const int IDX   = 1;
    static const int VAL   = 2;
    static const int END   = 3;

    uint32_t nbyte[END];
    int      nnz{0};
};

}  // namespace cusz

#endif
