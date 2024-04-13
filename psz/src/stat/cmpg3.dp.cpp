#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include "detail/ebness_thrust.inl"
#include "stat/compare/compare.dpl.hh"

#define THRUSTGPU_COMPARE_LOSSY(Tliteral, T)       \
    template bool psz::thrustgpu_error_bounded<T>( \
        T * a, T * b, size_t const len, double const eb, size_t* first_faulty_idx);

THRUSTGPU_COMPARE_LOSSY(fp32, float);
THRUSTGPU_COMPARE_LOSSY(fp64, double);

#undef THRUSTGPU_COMPARE_LOSSY
