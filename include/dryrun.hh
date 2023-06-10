#ifndef D3610824_7841_4292_99E9_D3F4F57E5C80
#define D3610824_7841_4292_99E9_D3F4F57E5C80

#include "context.h"
#include "utils/capsule.hh"
#include "utils2/memseg_cxx.hh"

namespace cusz {

template <typename T>
class Dryrunner {
   public:
    using BYTE = uint8_t;

   private:
    // Capsule<T> original;
    // Capsule<T> reconst;

    pszmem_cxx<T>* original;
    pszmem_cxx<T>* reconst;

   protected:
    // cusz_context* ctx;
    // int    dict_size;
    // dim3   xyz;
    double eb, max, min, rng;

   public:
    Dryrunner() = default;
    ~Dryrunner();

    Dryrunner& generic_dryrun(const std::string fname, double eb, int radius, bool r2r, cudaStream_t stream);
    Dryrunner& dualquant_dryrun(const std::string fname, double eb, bool r2r, cudaStream_t stream);

    Dryrunner& init_generic_dryrun(dim3 size);
    Dryrunner& destroy_generic_dryrun();
    Dryrunner& init_dualquant_dryrun(dim3 size);
    Dryrunner& destroy_dualquant_dryrun();
};

}  // namespace cusz

#endif /* D3610824_7841_4292_99E9_D3F4F57E5C80 */
