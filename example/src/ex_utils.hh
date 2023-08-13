#include <cstddef>
#include <cstdint>

template <typename T>
uint32_t count_outlier(T* in, size_t inlen, int radius, void* stream);