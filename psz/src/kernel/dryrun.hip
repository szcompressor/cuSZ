// deps
#include "cusz/type.h"
#include "dryrun.hh"
// definitions
#include "kernel/detail/dryrun_cuhip.inl"

template void psz::cuhip::GPU_lorenzo_dryrun(
    size_t len, f4* original, f4* reconst, double eb, void* stream);
template void psz::cuhip::GPU_lorenzo_dryrun(
    size_t len, f8* original, f8* reconst, double eb, void* stream);