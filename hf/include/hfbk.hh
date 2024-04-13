#ifndef CD8DAB1B_C057_43DB_A4F1_33653D9B545D
#define CD8DAB1B_C057_43DB_A4F1_33653D9B545D

#include <cstdint>

#include "cusz/type.h"

namespace psz {

template <pszpolicy P, typename E, typename H>
void hf_buildbook(
    uint32_t* freq, int const bklen, H* book, uint8_t* revbook,
    int const revbook_bytes, float* time, void* stream = nullptr);

}

#endif /* CD8DAB1B_C057_43DB_A4F1_33653D9B545D */
