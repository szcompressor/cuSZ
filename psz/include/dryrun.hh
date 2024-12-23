#ifndef D3610824_7841_4292_99E9_D3F4F57E5C80
#define D3610824_7841_4292_99E9_D3F4F57E5C80

#include "context.h"
#include "mem/cxx_memobj.h"
#include "port.hh"

namespace psz {
namespace cuhip {

template <typename T>
void GPU_lorenzo_dryrun(size_t len, T* original, T* reconst, PROPER_EB eb, void* stream);

}

namespace dpcpp {

template <typename T>
void GPU_lorenzo_dryrun(size_t len, T* original, T* reconst, PROPER_EB eb, void* stream);

}

}  // namespace psz

#endif /* D3610824_7841_4292_99E9_D3F4F57E5C80 */
