/**
 * @file lossless_huffman.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-06-16
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_LOSSLESS_HUFFMAN_H
#define CUSZ_WRAPPER_LOSSLESS_HUFFMAN_H

#define UNINITIALIZED unsigned char
#define BYTE unsigned char

template <typename Input, typename Huff, typename MetadataT = size_t>
struct HuffmanEncodingDescriptor {
    /****************************************************************************************************
     *                                         device  host
     *             type        length          space   space   archive description
     *
     * freq        uint32      len.dict        x       x               histogram frequency
     * book        Huff        len.dict        x       x               book for encoding
     * seg_uints   MetadataT   nchunk          x       x               segmental uint numbers (derived from seg_bits)
     * seg_bits    MetadataT   nchunk          x       x       x       segmental bit numbers
     * seg_entries MetadataT   nchunk          x       x       x       segmental entries (derived from seg_uints)
     * revbook     BYTE        len.revbook     x       x       x       reverse book for decoding
     * bitstream   BYTE        sum(seg_uints)  x       x       x       output bitsteram
     ****************************************************************************************************/
    static const size_t type_bitcount = sizeof(Huff) * 8;

    size_t       num_bits;   // analysis only
    unsigned int num_uints;  // part of metadata
    unsigned int nchunk;

    /********************************************************************************
     * pointer as accessor
     ********************************************************************************/
    struct {
        // codebook
        unsigned int* d_freq;     // .len() == dict_size
        Huff*         d_book;     // .len() == dict_size
        BYTE*         d_revbook;  // .len() == ... (see constructor)
        // encoding metadata
        MetadataT* d_seg_bits;     // .len() == nchunk
        MetadataT* d_seg_uints;    // .len() == nchunk
        MetadataT* d_seg_entries;  // .len() == nchunk
        // encoding (old method)
        Huff* fixed_len;    // .len() == original datalen; alloc otherwise
        Huff* d_bitstream;  // .len() == num_uints

        // codebook
        unsigned int* h_freq;     // .len() == dict_size (optional internal use)
        Huff*         h_book;     // .len() == dict_size (optional internal use)
        BYTE*         h_revbook;  // .len() == ... (see constructor)
        // encoding metadata
        MetadataT* h_seg_bits;     // .len() == nchunk
        MetadataT* h_seg_uints;    // .len() == nchunk
        MetadataT* h_seg_entries;  // .len() == nchunk
        // encoding (old method)
        Huff* h_bitstream;  // .len() == num_uints
    } space;

    /********************************************************************************
     * length and offset in numbers of byte ("raw")
     ********************************************************************************/
    struct {
        size_t freq, book, seg_uints;                      // host and dev space
        size_t seg_bits, seg_entries, revbook, bitstream;  // archive
    } rawlen;

    /********************************************************************************
     * `archive` is a subset of `space`
     ********************************************************************************/
    struct {
        BYTE*      h_revbook;      // .len() == ... (see constructor)
        MetadataT* h_seg_bits;     // .len() == nchunk
        MetadataT* h_seg_entries;  // .len() == nchunk

        /* According to expected CR, 1/2 original data is okay. */
        Huff* h_bitstream;  // .len() == num_uints
    } archive;

    /********************************************************************************
     * typed len
     ********************************************************************************/
    struct {
        unsigned int book;     // T = Huff
        unsigned int revbook;  // T = BYTE
        unsigned int freq;     // T = unsigned int
        unsigned int seg_;     // T = MetadataT
    } len;

    BYTE* d_space;
    BYTE* d_archive;  // reserved for GDS (gpu direct storage)
    BYTE* h_space;    //
    BYTE* h_archive;  // h_space + nbyte(seg_uints)

    HuffmanEncodingDescriptor(
        UNINITIALIZED* _d_space,
        UNINITIALIZED* _h_space,
        UNINITIALIZED* _h_archive,
        size_t         _dict_size,
        size_t         _nchunk) :
        d_space(_d_space), h_space(_h_space), h_archive(_h_archive), nchunk(_nchunk)
    {
        len.book    = _dict_size;
        len.freq    = _dict_size;
        len.revbook = sizeof(Huff) * (2 * this->type_bitcount) + sizeof(Input) * _dict_size;
        len.seg_    = nchunk;

        auto get_nbyte = [](auto ptr, size_t len) { return sizeof(std::remove_pointer<decltype(ptr)>::type) * len; };

        rawlen.freq        = _get_nbyte(space.d_freq, len.freq);
        rawlen.book        = _get_nbyte(space.d_book, len.book);
        rawlen.seg_uints   = _get_nbyte(space.d_seg_uints, len.seg_);
        rawlen.seg_bits    = _get_nbyte(space.d_seg_bitss, len.seg_);
        rawlen.seg_entries = _get_nbyte(space.d_seg_entries, len.seg_);
        rawlen.revbook     = _get_nbyte(space.d_revbook, len.revbook);
        rawlen.bitstream   = 0;  // set in process_huffman_metadata

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
        unsigned int& _rawlen_seg_bits,
        unsigned int& _rawlen_seg_entries,
        unsigned int& _rawlen_revbook,
        unsigned int& _rawlen_bitstream)
    {
        if (num_uints == 0) throw std::runtime_error("[ENC_CTX::save_metadata] num_units must not be 0.");

        rawlen.bitstream = num_uints * sizeof(Huff);

        _rawlen_seg_bits    = rawlen.seg_bits;
        _rawlen_seg_entries = rawlen.seg_entries;
        _rawlen_revbook     = rawlen.revbook;
        _rawlen_bitstream   = rawlen.bitstream;
    }
};

template <typename Input, typename Huff, typename MetadataT = size_t>
struct HuffmanDecodingDescriptor {
    /****************************************************************************************************
     *                                         device  host
     *             type        length          space   space   archive description
     *
     * seg_bits    MetadataT   nchunk          x       x       x       segmental bit numbers
     * seg_entries MetadataT   nchunk          x       x       x       segmental entries
     * revbook     BYTE        len.revbook     x       x       x       reverse book for decoding
     * bitstream   BYTE        sum(seg_uints)  x       x       x       output bitsteram
     ****************************************************************************************************/
    static const size_t type_bitcount = sizeof(Huff) * 8;

    unsigned int num_uints;
    unsigned int nchunk;
    struct {
        MetadataT* d_seg_uints;    // .len() == nchunk
        MetadataT* d_seg_entries;  // .len() == nchunk
        BYTE*      d_revbook;      // .len() == ... (see constructor)
        Huff*      d_bitstream;    // .len() == num_uints

        MetadataT* h_seg_uints;    // .len() == nchunk
        MetadataT* h_seg_entries;  // .len() == nchunk
        BYTE*      h_revbook;      // .len() == ... (see constructor)
        Huff*      h_bitstream;    // .len() == num_uints
    } space;

    struct {
        BYTE*      h_revbook;      // .len() = ... (see constructor)
        MetadataT* h_seg_uints;    // .len() == nchunk
        MetadataT* h_seg_entries;  // .len() == nchunk
        Huff*      h_bitstream;    // .len() == num_uints
    } archive;

    struct {
        size_t freq, book, seg_uints;                      // host and dev space
        size_t seg_bits, seg_entries, revbook, bitstream;  // archive
    } rawlen;
    struct {
        unsigned int book;     // T = Huff
        unsigned int revbook;  // T = BYTE
    } len;

    BYTE* d_archive;
    BYTE* h_archive;

    HuffmanDecodingDescriptor(
        BYTE*         _h_archive,
        size_t        _dict_size,
        size_t        _nchunk,
        unsigned int& _rawlen_bitstream,
        unsigned int& _rawlen_seg_bits,
        unsigned int& _rawlen_seg_entries,
        unsigned int& _rawlen_revbook

        ) :
        h_archive(_h_archive), nchunk(_nchunk)
    {
        this->len.cb      = _dict_size;
        this->len.revbook = sizeof(Huff) * (2 * this->type_bitcount) + sizeof(Input) * _dict_size;

        rawlen.seg_bits    = _rawlen_seg_bits;
        rawlen.seg_entries = _rawlen_seg_entries;
        rawlen.revbook     = _rawlen_revbook;
        rawlen.bitstream   = _rawlen_bitstream;
    }
};

template <typename Input, typename Huff, typename MetadataT = size_t, bool NSYMBOL_RESTRICT = true>
void compress_huffman_encode(HuffmanEncodingDescriptor<Input, Huff, MetadataT>*, Input*, size_t, int);

template <typename Output, typename Huff, typename MetadataT = size_t, bool NSYMBOL_RESTRICT = true>
void decompress_huffman_decode(HuffmanDecodingDescriptor<Output, Huff, MetadataT>*, Output*, size_t, int);

#endif
