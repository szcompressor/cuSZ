/**
 * @file definition.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-20
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_COMMON_DEFINITION_HH
#define CUSZ_COMMON_DEFINITION_HH

#include <cstdint>
#include <unordered_map>
#include <vector>
#include "../header.hh"

enum class cuszTASK { COMPRESS, DECOMPRESS, EXPERIMENT, COMPRESS_DRYRUN };
enum class cuszDEV { TEST, DEV, RELEASE };
enum class cuszLOC { HOST, DEVICE, HOST_DEVICE, UNIFIED, FS, NONE, __BUFFER };
enum class cuszWHEN { COMPRESS, DECOMPRESS, EXPERIMENT, COMPRESS_DRYRUN };
enum class ALIGNDATA { NONE, SQUARE_MATRIX, POWEROF2, NEXT_EVEN };
enum class ALIGNMEM { NONE, WARP32B, WARP64B, WARP128B };

struct Align {
    template <ALIGNDATA ad = ALIGNDATA::NONE>
    static size_t get_aligned_datalen(size_t len)
    {
        if CONSTEXPR (ad == ALIGNDATA::NONE) return len;
        if CONSTEXPR (ad == ALIGNDATA::SQUARE_MATRIX) {
            auto m = Reinterpret1DTo2D::get_square_size(len);
            return m * m;
        }
    }

    static const int DEFAULT_ALIGN_NBYTE = 128;

    template <int NUM>
    static inline bool is_aligned_at(const void* ptr)
    {  //
        return reinterpret_cast<uintptr_t>(ptr) % NUM == 0;
    };

    template <typename T, int NUM = DEFAULT_ALIGN_NBYTE>
    static size_t get_aligned_nbyte(size_t len)
    {
        return ((sizeof(T) * len - 1) / NUM + 1) * NUM;
    }
};

// TODO when to use ADDR8?
// TODO change to `enum class`
enum class cuszSEG { HEADER, BOOK, QUANT, REVBOOK, ANCHOR, SPFMT, HUFF_META, HUFF_DATA };

class DataSeg {
   public:
    std::unordered_map<cuszSEG, int> name2order = {
        {cuszSEG::HEADER, 0}, {cuszSEG::BOOK, 1},      {cuszSEG::QUANT, 2},     {cuszSEG::REVBOOK, 3},
        {cuszSEG::SPFMT, 4},  {cuszSEG::HUFF_META, 5}, {cuszSEG::HUFF_DATA, 6},  //
        {cuszSEG::ANCHOR, 7}};

    std::unordered_map<int, cuszSEG> order2name = {
        {0, cuszSEG::HEADER}, {1, cuszSEG::BOOK},      {2, cuszSEG::QUANT},     {3, cuszSEG::REVBOOK},
        {4, cuszSEG::SPFMT},  {5, cuszSEG::HUFF_META}, {6, cuszSEG::HUFF_DATA},  //
        {7, cuszSEG::ANCHOR}};

    std::unordered_map<cuszSEG, uint32_t> nbyte = {
        {cuszSEG::HEADER, sizeof(cusz_header)},
        {cuszSEG::BOOK, 0U},
        {cuszSEG::QUANT, 0U},
        {cuszSEG::REVBOOK, 0U},
        {cuszSEG::ANCHOR, 0U},
        {cuszSEG::SPFMT, 0U},
        {cuszSEG::HUFF_META, 0U},
        {cuszSEG::HUFF_DATA, 0U}};

    std::unordered_map<cuszSEG, std::string> name2str{
        {cuszSEG::HEADER, "HEADER"},       {cuszSEG::BOOK, "BOOK"},          {cuszSEG::QUANT, "QUANT"},
        {cuszSEG::REVBOOK, "REVBOOK"},     {cuszSEG::ANCHOR, "ANCHOR"},      {cuszSEG::SPFMT, "SPFMT"},
        {cuszSEG::HUFF_META, "HUFF_META"}, {cuszSEG::HUFF_DATA, "HUFF_DATA"}};

    std::vector<uint32_t> offset;

    uint32_t    get_offset(cuszSEG name) { return offset.at(name2order.at(name)); }
    std::string get_namestr(cuszSEG name) { return name2str.at(name); }
};

namespace cusz {
enum class execution { cuda, serial };
enum class method { native, thrust };

struct OK {
    template <cuszDEV m>
    static void ALLOC()
    {
        static_assert(
            m == cuszDEV::TEST or m == cuszDEV::DEV,  //
            "muse be cuszDEV::TEST or cuszDEV::DEV; use with caution");
    }

    template <cuszDEV m>
    static void FREE()
    {
        static_assert(
            m == cuszDEV::TEST or m == cuszDEV::DEV,  //
            "muse be cuszDEV::TEST or cuszDEV::DEV; use with caution");
    }
};

using ADDR4 = uint32_t;
using ADDR8 = size_t;

using FREQ = uint32_t;

};  // namespace cusz

#endif
