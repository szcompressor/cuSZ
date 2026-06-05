// Author: Jiannan Tian

#include <curand.h>

#include "utils/synth.hh"

template <>
void _portable::testutils::rand_array_cu<float>(
    float* array_g, size_t len, uint32_t seed)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, (unsigned long)seed);
  curandGenerateUniform(gen, array_g, len);
}

template <>
void _portable::testutils::rand_array_cu<double>(
    double* array_g, size_t len, uint32_t seed)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, (unsigned long)seed);
  curandGenerateUniformDouble(gen, array_g, len);
}
