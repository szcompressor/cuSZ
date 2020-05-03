#if not defined(PARSER_HH)
#define PARSER_HH

#include <boost/program_options.hpp>
#include <cstddef>
#include <string>
#include "types.hh"

namespace po = boost::program_options;
typedef po::options_description desc_t;

const size_t maxDim = 4;

typedef struct Context {
    std::string input_file, data_type;

    config_t*  C;
    dim_t*     D;
    psegSize_t* P;

    bool run_native_cuda;
    bool run_kokkos, run_kokkos_seq, run_kokkos_omp, run_kokkos_cuda;
    bool run_raja, run_raja_seq, run_raja_simd, run_raja_omp, run_raja_omp_gpu, run_raja_cuda;

    bool compress, decompress;

    Context();
    void debug();

} ctx_t;

int parser(int ac, char** av, ctx_t* ctx, desc_t& desc);

#endif