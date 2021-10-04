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

enum class cuszTASK { COMPRESS, DECOMPRESS, EXPERIMENT, COMPRESS_DRYRUN };
enum class cuszDEV { TEST, DEV, RELEASE };
enum class cuszLOC { HOST, DEVICE, HOST_DEVICE, UNIFIED, FS, NONE, __BUFFER };
enum class cuszWHEN { COMPRESS, DECOMPRESS, EXPERIMENT, COMPRESS_DRYRUN };

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