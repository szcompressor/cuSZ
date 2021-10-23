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

enum class cuszTASK { COMPRESS, DECOMPRESS, EXPERIMENT, COMPRESS_DRYRUN };
enum class cuszDEV { TEST, DEV, RELEASE };
enum class cuszLOC { HOST, DEVICE, HOST_DEVICE, UNIFIED, FS, NONE, __BUFFER };
enum class cuszWHEN { COMPRESS, DECOMPRESS, EXPERIMENT, COMPRESS_DRYRUN };
enum class ALIGNDATA { NONE, SQUARE_MATRIX, POWEROF2, NEXT_EVEN };
enum class ALIGNMEM { NONE, WARP32B, WARP64B, WARP128B };

struct cuszCOMPONENTS {
    struct PREDICTOR {
        static const uint32_t LORENZO   = 0;
        static const uint32_t LORENZOII = 1;
        static const uint32_t SPLINE3   = 2;
    };
    struct CODEC {
        static const uint32_t HUFFMAN_COARSE = 0;
    };
    struct SPREDUCER {
        static const uint32_t CSR11 = 0;
    };
};

// TODO when to use ADDR8?
// TODO change to `enum class`
enum class cuszSEG { HEADER, BOOK, QUANT, REVBOOK, ANCHOR, SPFMT, HUFF_META, HUFF_DATA };

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
