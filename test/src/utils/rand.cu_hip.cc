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

#if defined(PSZ_USE_CUDA)

#include <curand.h>

#elif defined(PSZ_USE_HIP)

#include <hiprand/hiprand.h>

#endif

#include "../rand.hh"

template <>
void psz::testutils::cu_hip::rand_array<float>(
    float* array_g, size_t len, uint32_t seed)
{
#if defined(PSZ_USE_CUDA)

  curandGenerator_t gen;
  // Create a new generator
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  // Set the generator options
  curandSetPseudoRandomGeneratorSeed(gen, (unsigned long)seed);
  // Generate random numbers
  curandGenerateUniform(gen, array_g, len);

#elif defined(PSZ_USE_HIP)

  hiprandGenerator_t gen;
  // Create a new generator
  hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
  // Set the generator options
  hiprandSetPseudoRandomGeneratorSeed(gen, (unsigned long)seed);
  // Generate random numbers
  hiprandGenerateUniform(gen, array_g, len);

#endif
}

template <>
void psz::testutils::cu_hip::rand_array<double>(
    double* array_g, size_t len, uint32_t seed)
{
#if defined(PSZ_USE_CUDA)

  curandGenerator_t gen;
  // Create a new generator
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  // Set the generator options
  curandSetPseudoRandomGeneratorSeed(gen, (unsigned long)seed);
  // Generate random numbers
  curandGenerateUniformDouble(gen, array_g, len);

#elif defined(PSZ_USE_HIP)

  hiprandGenerator_t gen;
  // Create a new generator
  hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
  // Set the generator options
  hiprandSetPseudoRandomGeneratorSeed(gen, (unsigned long)seed);
  // Generate random numbers
  hiprandGenerateUniformDouble(gen, array_g, len);

#endif
}