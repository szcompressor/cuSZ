#ifndef VERIFY_HH
#define VERIFY_HH

#include <cstddef>
#include <cstdint>

namespace analysis {

template <typename T>
void VerifyData(T* xData, T* oData, size_t _len, bool override_eb = false, double new_eb = 0, size_t archive_byte_size = 0, size_t binning_scale = 1);

}  // namespace analysis

#endif
