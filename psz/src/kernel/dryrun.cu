// deps
#include "cusz/type.h"
#include "dryrun.hh"
#include "port.hh"
// definitions
#include "kernel/detail/dryrun.cu_hip.inl"

template void psz::cu_hip::dryrun(
    size_t len, f4* original, f4* reconst, double eb, void* stream);
template void psz::cu_hip::dryrun(
    size_t len, f8* original, f8* reconst, double eb, void* stream);