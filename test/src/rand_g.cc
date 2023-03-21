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

#include <cuda.h>
#include <curand.h>
#include "rand.hh"

template <>
void psz::testutils::cuda::rand_array<float>(float* array_g, size_t len)
{
    curandGenerator_t gen;

    // Create a new generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the generator options
    curandSetPseudoRandomGeneratorSeed(gen, (unsigned long)1234ULL);
    // Generate random numbers
    curandGenerateUniform(gen, array_g, len);
}

template <>
void psz::testutils::cuda::rand_array<double>(double* array_g, size_t len)
{
    curandGenerator_t gen;

    // Create a new generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the generator options
    curandSetPseudoRandomGeneratorSeed(gen, (unsigned long)1234ULL);
    // Generate random numbers
    curandGenerateUniformDouble(gen, array_g, len);
}