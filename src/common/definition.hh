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

namespace cusz {

enum class MODE { TEST, DEV, RELEASE };
enum class WHERE { HOST, DEVICE, BOTH, FS };
enum class execution { cuda, serial };
enum class method { native, thrust };
enum class WHEN { COMPRESS, DECOMPRESS, EXPERIMENT, COMPRESS_DRYRUN };

struct OK {
    template <MODE m>
    constexpr bool ALLOCATE()
    {
        return m == MODE::TEST or m == MODE::DEV;
    }
};

using ADDR4 = uint32_t;
using ADDR8 = size_t;

using FREQ = uint32_t;

};  // namespace cusz

#endif