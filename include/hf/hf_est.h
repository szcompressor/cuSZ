#ifdef __cplusplus

extern "C" {
#endif

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void pszanalysis_hf_buildtree(
    uint32_t* freq, int const bklen, double* entropy, double* cr,
    int const symbol_byte);

#ifdef __cplusplus
}
#endif