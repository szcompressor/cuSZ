#ifndef D2F48D60_CCE7_4049_8A56_2ADDF140192E
#define D2F48D60_CCE7_4049_8A56_2ADDF140192E

#include <cstddef>
#include <cstdint>

#include "cusz/type.h"
#include "kernel/hist.hh"
#include "kernel/predictor.hh"

template <typename T>
uint32_t count_outlier(T* in, size_t inlen, int radius, void* stream);

#endif /* D2F48D60_CCE7_4049_8A56_2ADDF140192E */
