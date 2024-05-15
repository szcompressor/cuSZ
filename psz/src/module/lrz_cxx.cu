#include "lrz_cxx.cu_hip.inl"

template class _2401::pszpred_lrz<f4, u2, CPU_BARRIER_AND_TIMING>;
template class _2401::pszpred_lrz<f4, u2, CPU_BARRIER>;
template class _2401::pszpred_lrz<f4, u2, GPU_AUTOMONY>;

template class _2401::pszpred_lrz<f8, u2, CPU_BARRIER_AND_TIMING>;
template class _2401::pszpred_lrz<f8, u2, CPU_BARRIER>;
template class _2401::pszpred_lrz<f8, u2, GPU_AUTOMONY>;

template class _2401::pszpred_lrz<f4, u4, CPU_BARRIER_AND_TIMING>;
template class _2401::pszpred_lrz<f4, u4, CPU_BARRIER>;
template class _2401::pszpred_lrz<f4, u4, GPU_AUTOMONY>;

template class _2401::pszpred_lrz<f8, u4, CPU_BARRIER_AND_TIMING>;
template class _2401::pszpred_lrz<f8, u4, CPU_BARRIER>;
template class _2401::pszpred_lrz<f8, u4, GPU_AUTOMONY>;

#undef INS