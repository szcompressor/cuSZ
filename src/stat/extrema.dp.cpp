#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cusz/type.h"
#include "stat/compare/compare.dp.hh"
#include "utils/err.hh"
#include "port.hh"
// definitions
#include "detail/extrema.dp.inl"

template void psz::dpcpp::extrema<f4>(f4* d_ptr, szt len, f4 res[4]);
// template void psz::dpcpp::extrema<f8>(f8* d_ptr, szt len, f8 res[4]);