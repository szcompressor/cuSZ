
#include "../rand.hh"

#include <dpct/dpct.hpp>
#include <dpct/rng_utils.hpp>
#include <sycl/sycl.hpp>

template <>
void psz::testutils::dpcpp::rand_array<float>(
    float* array_g, size_t len, uint32_t seed)
{
  dpct::rng::host_rng_ptr gen;
  // Create a new generator
  gen = dpct::rng::create_host_rng(dpct::rng::random_engine_type::mcg59);
  // Set the generator options
  gen->set_seed((unsigned long)seed);
  // Generate random numbers
  gen->generate_uniform(array_g, len);
}

template <>
void psz::testutils::dpcpp::rand_array<double>(
    double* array_g, size_t len, uint32_t seed)
{
  dpct::rng::host_rng_ptr gen;
  // Create a new generator
  gen = dpct::rng::create_host_rng(dpct::rng::random_engine_type::mcg59);
  // Set the generator options
  gen->set_seed((unsigned long)seed);
  // Generate random numbers
  gen->generate_uniform(array_g, len);
}