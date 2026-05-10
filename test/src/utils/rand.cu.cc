// Author: Jiannan Tian

#include <curand.h>

#include "../rand.hh"

template <>
void psz::testutils::cu_hip::rand_array<float>(
    float* array_g, size_t len, uint32_t seed)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, (unsigned long)seed);
  curandGenerateUniform(gen, array_g, len);
}

template <>
void psz::testutils::cu_hip::rand_array<double>(
    double* array_g, size_t len, uint32_t seed)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, (unsigned long)seed);
  curandGenerateUniformDouble(gen, array_g, len);
}