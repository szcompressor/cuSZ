/**
 * @file rand_g.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-21
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

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