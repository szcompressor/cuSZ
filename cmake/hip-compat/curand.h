// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Shim placed on the HIP include path so sources that #include <curand.h>
// resolve against hipRAND. The cuRAND host-generator API used by cuSZ
// (portable/src/utils/rand.cu.cc) maps 1:1 onto hipRAND.
#ifndef PSZ_HIP_COMPAT_CURAND_H
#define PSZ_HIP_COMPAT_CURAND_H

#include <hiprand/hiprand.h>

#define curandGenerator_t hiprandGenerator_t
#define curandCreateGenerator hiprandCreateGenerator
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandGenerateUniform hiprandGenerateUniform
#define curandGenerateUniformDouble hiprandGenerateUniformDouble
#define CURAND_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT

#endif
