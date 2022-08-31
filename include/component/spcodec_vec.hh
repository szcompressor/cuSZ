/**
 * @file spcodec_vec.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef CF358238_3946_4FFC_B5E6_45C12F0C0B44
#define CF358238_3946_4FFC_B5E6_45C12F0C0B44

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>

#define DEFINE_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};

namespace cusz {

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

#endif /* CF358238_3946_4FFC_B5E6_45C12F0C0B44 */
