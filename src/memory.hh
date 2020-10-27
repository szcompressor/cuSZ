/**
 * @file memory.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.2
 * @date 2020-10-25
 *
 * (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>

namespace internal {

struct allocator {
    // Allocates on the current device!
    void* operator()(size_t num_bytes) const { return detail::allocate(num_bytes).start; }
};

struct deleter {
    void operator()(void* ptr) { cudaFree(ptr); }
};
}  // namespace internal