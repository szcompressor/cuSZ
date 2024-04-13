// deps
#include "dryrun.hh"

#include "cusz/type.h"
#include "port.hh"
// definitions
#include "kernel/detail/dryrun.dp.inl"

template void psz::dpcpp::dryrun(
    size_t len, f4* original, f4* reconst, PROPER_EB eb, void* stream);
// template void psz::dpcpp::dryrun(
//     size_t len, f8* original, f8* reconst, PROPER_EB eb, void* stream);