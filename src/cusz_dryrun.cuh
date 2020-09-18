// jtian 20-05-14

#include <string>

namespace cusz {

namespace dryrun {
template <typename T>
__global__ void lorenzo_1d1l(T*, size_t*, double*);

template <typename T>
__global__ void lorenzo_2d1l(T*, size_t*, double*);

template <typename T>
__global__ void lorenzo_3d1l(T*, size_t*, double*);

}  // namespace dryrun

namespace workflow {

template <typename T>
void DryRun(T*, T*, std::string, size_t*, double*);

}
}  // namespace cusz
