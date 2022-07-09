/**
 * @file codec.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_COMPONENT_CODECS_HH
#define CUSZ_COMPONENT_CODECS_HH

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>

#define DEFINE_ARRAY(VAR, TYPE) \
    TYPE* d_##VAR{nullptr};     \
    TYPE* h_##VAR{nullptr};

namespace cusz {

// template <typename T, typename H, typename M>
// class CodecInterface {
//    public:
//     virtual float get_time_elapsed() const  = 0;
//     virtual float get_time_book() const     = 0;
//     virtual float get_time_lossless() const = 0;
//     virtual void  clear_buffer()            = 0;
// };

template <typename T, typename H, typename M>
class LosslessCodec
// : CodecInterface<T, H, M>
{
   public:
    using Origin    = T;
    using Encoded   = H;
    using MetadataT = M;
    using FreqT     = uint32_t;
    using BYTE      = uint8_t;

   private:
    class impl;
    std::unique_ptr<impl> pimpl;

   public:
    ~LosslessCodec();                                // dtor
    LosslessCodec();                                 // ctor
    LosslessCodec(const LosslessCodec&);             // copy ctor
    LosslessCodec& operator=(const LosslessCodec&);  // copy assign
    LosslessCodec(LosslessCodec&&);                  // move ctor
    LosslessCodec& operator=(LosslessCodec&&);       // move assign

    void init(size_t const, int const, int const, bool dbg_print = false);
    void build_codebook(uint32_t*, int const, cudaStream_t = nullptr);
    void encode(T*, size_t const, uint32_t*, int const, int const, int const, BYTE*&, size_t&, cudaStream_t = nullptr);
    void decode(BYTE*, T*, cudaStream_t = nullptr, bool = true);
    void clear_buffer();

    float get_time_elapsed() const;
    float get_time_book() const;
    float get_time_lossless() const;
};

template <typename T, typename H, typename M>
class LosslessCodec<T, H, M>::impl {
   public:
    using Origin    = T;
    using Encoded   = H;
    using MetadataT = M;
    using FreqT     = uint32_t;
    using BYTE      = uint8_t;

   private:
    using BOOK = H;
    using SYM  = T;

    // TODO shared header
    struct Header {
        static const int HEADER    = 0;
        static const int REVBOOK   = 1;
        static const int PAR_NBIT  = 2;
        static const int PAR_ENTRY = 3;
        static const int BITSTREAM = 4;
        static const int END       = 5;

        int       header_nbyte : 16;
        int       booklen : 16;
        int       sublen;
        int       pardeg;
        size_t    uncompressed_len;
        size_t    total_nbit;
        size_t    total_ncell;  // TODO change to uint32_t
        MetadataT entry[END + 1];

        MetadataT subfile_size() const { return entry[END]; }
    };

    struct runtime_encode_helper {
        static const int TMP       = 0;
        static const int FREQ      = 1;
        static const int BOOK      = 2;
        static const int REVBOOK   = 3;
        static const int PAR_NBIT  = 4;
        static const int PAR_NCELL = 5;
        static const int PAR_ENTRY = 6;
        static const int BITSTREAM = 7;
        static const int END       = 8;

        uint32_t nbyte[END];
    };

    using RTE    = runtime_encode_helper;
    using Header = struct Header;

   private:
    // array
    DEFINE_ARRAY(tmp, H);
    DEFINE_ARRAY(compressed, BYTE);  // alias in address
    DEFINE_ARRAY(book, H);
    DEFINE_ARRAY(revbook, BYTE);

    DEFINE_ARRAY(par_metadata, M);
    DEFINE_ARRAY(par_nbit, M);
    DEFINE_ARRAY(par_ncell, M);
    DEFINE_ARRAY(par_entry, M);

    DEFINE_ARRAY(bitstream, H);
    // helper
    RTE rte;
    // memory
    static const int CELL_BITWIDTH = sizeof(H) * 8;
    // timer
    float milliseconds{0.0};
    float time_hist{0.0}, time_book{0.0}, time_lossless{0.0};

   public:
    ~impl();  // dtor
    impl();   // ctor

    // getter
    float         get_time_elapsed() const;
    float         get_time_book() const;
    float         get_time_lossless() const;
    size_t        get_workspace_nbyte(size_t) const;
    size_t        get_max_output_nbyte(size_t len) const;
    static size_t get_revbook_nbyte(int);
    // getter for internal array
    H*    expose_book() const;
    BYTE* expose_revbook() const;
    // compile-time
    constexpr bool can_overlap_input_and_firstphase_encode();
    // public methods
    void init(size_t const, int const, int const, bool dbg_print = false);
    void build_codebook(uint32_t*, int const, cudaStream_t = nullptr);
    void encode(T*, size_t const, uint32_t*, int const, int const, int const, BYTE*&, size_t&, cudaStream_t = nullptr);
    void decode(BYTE*, T*, cudaStream_t = nullptr, bool = true);
    void clear_buffer();

   private:
    void subfile_collect(Header&, size_t const, int const, int const, int const, cudaStream_t stream = nullptr);
    void dbg_println(const std::string, void*, int);
};

}  // namespace cusz

#undef DEFINE_ARRAY

#endif
