/**
 * @file huffman_coarse.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-18
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_HUFFMAN_COARSE_CUH
#define CUSZ_WRAPPER_HUFFMAN_COARSE_CUH

#include <cstdint>

#include "../../include/reducer.hh"
#include "../common/definition.hh"
#include "../common/type_traits.hh"
#include "../header.hh"

namespace cusz {

template <typename T, typename H, typename M = uint64_t>
class HuffmanWork {
   public:
    using Origin  = T;
    using Encoded = H;
    using Mtype   = M;

   private:
    using BYTE      = uint8_t;
    using ADDR_OFST = uint32_t;

    cuszHEADER* header;
    BYTE*       dump;
    BYTE*       h_revbook;
    H*          h_bitstream;
    M*          h_bits_entries;

    float milliseconds;

    uint32_t orilen;
    uint32_t nchunk, chunk_size, num_uints, revbook_nbyte;

   public:
    //
    float get_time_elapsed() const { return milliseconds; }

    //
    HuffmanWork(cuszHEADER* _header, uint32_t _orilen)
    {
        header = _header;
        orilen = _orilen;
    }
    HuffmanWork(uint32_t _orilen, BYTE* _dump, uint32_t _chunk_size, uint32_t _num_uints, uint32_t _dict_size)
    {
        dump          = _dump;
        orilen        = _orilen;
        chunk_size    = _chunk_size;
        nchunk        = ConfigHelper::get_npart(orilen, chunk_size);
        num_uints     = _num_uints;
        revbook_nbyte = HuffmanHelper::get_revbook_nbyte<T, H>(_dict_size);
    }

    ~HuffmanWork() {}

    void decode(H* in, T* out);

    void decode(cusz::LOC loc, H* bitstream, M* bits_entries, BYTE* revbook, T* out);
};

}  // namespace cusz

#endif