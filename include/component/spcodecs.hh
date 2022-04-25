/**
 * @file spcodecs.hh
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

#include <cstdint>

namespace cusz {
namespace api {

template <typename T, typename M = uint32_t>
class SpCodecCSR {
   public:
    using Origin = T;
    using BYTE   = uint8_t;

   public:
    struct impl;

   public:
    SpCodecCSR() = default;
    ~SpCodecCSR();
    void init(size_t const, int = 4, bool = false);
    void encode(T*, size_t const, BYTE*&, size_t&, cudaStream_t = nullptr, bool = false);
    void decode(BYTE*, T*, cudaStream_t = nullptr, bool = true);
    void clear_buffer();
    // getter
    float get_time_elapsed() const;
};

template <typename T, typename M>
struct SpCodecCSR<T, M>::impl {
   public:
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
    void decode(BYTE*, T*, cudaStream_t = nullptr, bool = true);
    void clear_buffer();
    // getter
    float get_time_elapsed() const;
};

template <typename T, typename M = uint32_t>
class SpCodecVec {
   public:
    using Origin = T;
    using BYTE   = uint8_t;

   public:
    struct impl;

   public:
    SpCodecVec() = default;
    ~SpCodecVec();
    void init(size_t const, int = 4, bool = false);
    void encode(T*, size_t const, BYTE*&, size_t&, cudaStream_t = nullptr, bool = false);
    void decode(BYTE*, T*, cudaStream_t = nullptr, bool = true);
    void clear_buffer();
    // getter
    float get_time_elapsed() const;
};

template <typename T, typename M>
struct SpCodecVec<T, M>::impl {
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

}  // namespace api
}  // namespace cusz

#endif
