#include "max_err.cuhip.inl"

template void psz::module::GPU_find_max_error<float>(
    float* a, float* b, size_t const len, float& maxval, size_t& maxloc, void* stream);

template void psz::module::GPU_find_max_error<double>(
    double* a, double* b, size_t const len, double& maxval, size_t& maxloc, void* stream);