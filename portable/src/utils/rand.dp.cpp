// Author: Jiannan Tian

#include <dpct/dpct.hpp>
#include <dpct/rng_utils.hpp>
#include <sycl/sycl.hpp>

#include "utils/synth.hh"

template <>
void _ptb::testutils::rand_array_dpcpp<float>(float* array_g, size_t len, uint32_t seed)
{
  dpct::rng::host_rng_ptr gen;
  gen = dpct::rng::create_host_rng(dpct::rng::random_engine_type::mcg59);
  gen->set_seed((unsigned long)seed);
  gen->generate_uniform(array_g, len);
}

template <>
void _ptb::testutils::rand_array_dpcpp<double>(double* array_g, size_t len, uint32_t seed)
{
  dpct::rng::host_rng_ptr gen;
  gen = dpct::rng::create_host_rng(dpct::rng::random_engine_type::mcg59);
  gen->set_seed((unsigned long)seed);
  gen->generate_uniform(array_g, len);
}
