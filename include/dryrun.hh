#ifndef D3610824_7841_4292_99E9_D3F4F57E5C80
#define D3610824_7841_4292_99E9_D3F4F57E5C80

#include "context.h"
#include "mem/memseg_cxx.hh"

namespace psz {
namespace cu_hip {

template <typename T>
void dryrun(size_t len, T* original, T* reconst, double eb, void* stream);

}

namespace dpcpp {

template <typename T>
void dryrun(size_t len, T* original, T* reconst, double eb, void* stream);

}

}  // namespace psz

namespace psz {

template <typename T>
void dryrun(
    const std::string fname, size_t x, size_t y, size_t z, double eb, bool r2r,
    void* stream);
}

#endif /* D3610824_7841_4292_99E9_D3F4F57E5C80 */
