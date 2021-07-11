/**
 * @file lossless_huffman.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-07-10
 * (created) 2020-04-24 (rev.1) 2020-10-24 (rev.2) 2021-07-10
 *
 * (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_LOSSLESS_HUFFMAN_H
#define CUSZ_WRAPPER_LOSSLESS_HUFFMAN_H

#define UNINITIALIZED unsigned char
#define BYTE unsigned char

template <typename T>
struct PartialData {
    using type = T;
    T*           dptr;
    T*           hptr;
    unsigned int len;
    unsigned int nbyte() const {return len * sizeof(T)};
    void         h2d() { cudaMemcpy(dptr, hptr, nbytes(), cudaMemcpyHostToDevice); }
    void         d2h() { cudaMemcpy(hptr, dptr, nbytes(), cudaMemcpyDeviceToHost); }
};

template <typename Input, typename Huff, typename MetadataT = size_t>
struct HuffmanEncodingDescriptor {
    /****************************************************************************************************
     *                                         device  host   host
     *             type        length          space   space  archive description
     *
     * freq        uint32      len.dict        x       x              histogram frequency
     * book        Huff        len.dict        x       x              book for encoding
     * seg_uints   MetadataT   nchunk          x       x              segmental uint numbers (derived from seg_bits)
     * seg_bits    MetadataT   nchunk          x       x      x       segmental bit numbers
     * seg_entries MetadataT   nchunk          x       x      x       segmental entries (derived from seg_uints)
     * revbook     BYTE        len.revbook     x       x      x       reverse book for decoding
     * bitstream   uint32/64   sum(seg_uints)  x       x      x       output bitsteram
     ****************************************************************************************************/
    static const size_t type_bitcount = sizeof(Huff) * 8;

    size_t       num_bits;   // analysis only
    unsigned int num_uints;  // part of metadata
    unsigned int nchunk;
    unsigned int dict_size;

    struct {
        struct {
            struct PartialData<BYTE> uninitialized;

            struct PartialData<unsigned int> freq;
            struct PartialData<Huff>         book;
            struct PartialData<MetadataT>    seg_uints;
        } non_archive;

        struct {
            struct PartialData<BYTE> uninitialized;

            struct PartialData<MetadataT> seg_bits;
            struct PartialData<MetadataT> seg_entries;
            struct PartialData<BYTE>      revbook;
            struct PartialData<Huff>      bitstream;
        } archive;

        struct PartialData<input> input;
        struct PartialData<Huff>  fixed_len;

        //        size_t maximum_possible(bool nbyte) const {}
        //        size_t archive_size(bool nbyte) const {}
    } space;

    unsigned int query_revbook_len() const { return sizeof(Huff) * (2 * type_bitcount) + sizeof(Input) * dict_size; }

    unsigned int query_non_archive_nbyte() const
    {
        return space.non_archive.seg_uints.nbyte() +  //
               space.non_archive.book.nbyte() +       //
               space.non_archive.freq.nbyte();        //
    }

    unsigned int query_archive_nbyte() const
    {
        return space.archive.seg_bits.nbyte() +     //
               space.archive.seg_entries.nbyte() +  //
               space.archive.revbook.nbyte() +      //
               space.archive.bitstream.nbyte();
    }

    size_t query_pool_nbyte() const { return query_non_archive_nbyte() + query_archive_nbyte(); }

    unsigned int update_bitstream_size() { space.archive.bitstream.len = num_bits; }

    BYTE* d_space;
    BYTE* d_non_archive;
    BYTE* d_archive;  // reserved for GDS (gpu direct storage)
    BYTE* h_space;    //
    BYTE* h_archive;  // h_space + nbyte(seg_uints)

    HuffmanEncodingDescriptor(unsigned int _input_size, unsigned int _dict_size, unsigned int _nchunk)
    {
        auto get_nbyte = [](auto ptr, size_t len) { return sizeof(std::remove_pointer<decltype(ptr)>::type) * len; };

        nchunk = _nchunk;

        space.input         = {nullptr, nullptr, _input_size};
        space.fixed_len.len = _input_size;

        space.non_archive.freq.len      = _dict_size;
        space.non_archive.book.len      = _dict_size;
        space.non_archive.seg_uints.len = nchunk;
        space.archive.seg_bits.len      = nchunk;
        space.archive.seg_entries.len   = nchunk;
        space.archive.revbook.len       = query_revbook_len();
        space.archive.bitstream.len     = 0;
    }

    void configure(UNINITIALIZED* _d_space, UNINITIALIZED* _h_space)
    {
        d_space = _d_space;
        h_space = _h_space;

        d_archive = d_space;
        // h_archive = _h_archive;

        auto typify = [](auto ptr, auto raw_ptr, size_t raw_offset) {
            ptr = reinterpret_cast<decltype(ptr)>(raw_ptr + raw_offset);
        };
        auto typify_dspace = [&](auto ptr, size_t raw_offset) { typify(ptr, d_space, raw_offset); };
        auto typify_hspace = [&](auto ptr, size_t raw_offset) { typify(ptr, h_space, raw_offset); };
        auto typify_harchi = [&](auto ptr, size_t raw_offset) { typify(ptr, h_archive, raw_offset); };

        // clang-format off
        typify_dspace(space.d_freq,        0);
        typify_dspace(space.d_book,        rawlen.freq);
        typify_dspace(space.d_seg_uints,   rawlen.freq + rawlen.book);
        typify_dspace(space.d_seg_bits,    rawlen.freq + rawlen.book + rawlen.seg_uints);
        typify_dspace(space.d_seg_entries, rawlen.freq + rawlen.book + rawlen.seg_uints + rawlen.seg_bits);
        typify_dspace(space.d_revbook,     rawlen.freq + rawlen.book + rawlen.seg_uints + rawlen.seg_bits + rawlen.seg_entries);
        typify_dspace(space.d_fixed_len,   rawlen.freq + rawlen.book + rawlen.seg_uints + rawlen.seg_bits + rawlen.seg_entries + rawlen.revbook);
        typify_dspace(space.d_bitstream,   rawlen.freq + rawlen.book + rawlen.seg_uints + rawlen.seg_bits + rawlen.seg_entries + rawlen.revbook);

        typify_hspace(space.h_freq,        0);
        typify_hspace(space.h_book,        rawlen.freq);
        typify_hspace(space.h_seg_uints,   rawlen.freq + rawlen.book);

        typify_harchi(space.h_seg_bits,    0);
        typify_harchi(space.h_seg_entries, rawlen.seg_bits);
        typify_harchi(space.h_revbook,     rawlen.seg_bits + rawlen.seg_entries);
        typify_harchi(space.h_bitstream,   rawlen.seg_bits + rawlen.seg_entries + rawlen.revbook);
        // clang-format on
    }

    void save_metadata(
        unsigned int& len_seg_bits,
        unsigned int& len_seg_entries,
        unsigned int& nbyte_revbook,
        unsigned int& len_bitstream)
    {
        if (num_uints == 0) throw std::runtime_error("[ENC_CTX::save_metadata] num_units must not be 0.");

        len_seg_bits    = space.archive.seg_bits.len;
        len_seg_entries = space.archive.seg_entries.len;
        nbyte_revbook   = space.archive.revbook.len;
        len_bitstream   = space.archive.bitstream.len;
    }
};

template <typename Input, typename Huff, typename MetadataT = size_t>
struct HuffmanDecodingDescriptor {
    /****************************************************************************************************
     *                                         device  host
     *             type        length          space   space  description
     *
     * seg_bits    MetadataT   nchunk          x       x      segmental bit numbers
     * seg_entries MetadataT   nchunk          x       x      segmental entries
     * revbook     BYTE        len.revbook     x       x      reverse book for decoding
     * bitstream   uint32/64   sum(seg_uints)  x       x      output bitsteram
     ****************************************************************************************************/
    static const size_t type_bitcount = sizeof(Huff) * 8;

    unsigned int num_uints;
    unsigned int nchunk;
    unsigned int dict_size;

    // Listed variables follow the order in memory layout.
    struct {
        struct PartialData<BYTE> uninitialized;

        struct PartialData<MetadataT> seg_bits;
        struct PartialData<MetadataT> seg_entries;
        struct PartialData<BYTE>      revbook;
        struct PartialData<Huff>      bitstream;

    } space;

    BYTE* d_archive;
    BYTE* h_archive;

    unsigned int query_revbook_len() const { return sizeof(Huff) * (2 * type_bitcount) + sizeof(Input) * dict_size; }

    HuffmanDecodingDescriptor(BYTE* _h_archive, size_t _dict_size, size_t _nchunk, unsigned int num_uints) :
        h_archive(_h_archive), nchunk(_nchunk)
    {
        space.seg_bits    = {nullptr, nullptr, nchunk};
        space.seg_entries = {nullptr, nullptr, nchunk};
        space.revbook     = {nullptr, nullptr, query_revbook_len()};
        space.bitstream   = {nullptr, nullptr, num_uints};
    }
};

template <typename Input, typename Huff, typename MetadataT = size_t, bool NSYMBOL_RESTRICT = true>
void compress_huffman_encode(HuffmanEncodingDescriptor<Input, Huff, MetadataT>*, Input*, size_t, int);

template <typename Output, typename Huff, typename MetadataT = size_t, bool NSYMBOL_RESTRICT = true>
void decompress_huffman_decode(HuffmanDecodingDescriptor<Output, Huff, MetadataT>*, Output*, size_t, int);

#endif
