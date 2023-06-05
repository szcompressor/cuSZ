/**
 * @file hf_struct.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-09-14
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef DA6883A3_A70F_4690_A4FA_56644987725A
#define DA6883A3_A70F_4690_A4FA_56644987725A

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

#include "cusz/type.h"
#include "layout.h"

// raw pointer array; regardless of being on host or device
typedef struct hf_book {
    uint32_t* freq;
    // undertermined on definition; could be uint32_t* and uint64_t*
    void* book;
    int   booklen;
} hf_book;

// typedef struct hf_revbook {
// } hf_revbook;

typedef struct hf_chunk {
    void* bits;     // how many bits each chunk
    void* cells;    // how many cells each chunk
    void* entries;  // jump to the chunk
} hf_chunk;

typedef struct hf_bitstream {
    void*     buffer;
    void*     bitstream;
    hf_chunk* d_metadata;
    hf_chunk* h_metadata;
    int       sublen;  // data chunksize
    int       pardeg;  // runtime paralleism degree
    int       numSMs;  // number of streaming multiprocessor
} hf_bitstream;

// used on host
typedef struct hf_context {
    int booklen, sublen, pardeg;

    float _time_book, time_lossless;

    uint32_t* freq;

    hf_book*      book_desc;
    hf_bitstream* bitstream_desc;

    size_t total_nbit, total_ncell;
} hf_context;

typedef hf_context hf_ctx;

typedef struct hf_runtime_mem_layout {
    // device
    psz_memseg tmp;  // can overlap with err-quant array
    psz_memseg book;
    psz_memseg revbook;
    psz_memseg par_nbit;
    psz_memseg par_ncell;
    psz_memseg par_entry;
    psz_memseg bitstream;
    // host
    psz_memseg h_par_nbit;
    psz_memseg h_par_ncell;
    psz_memseg h_par_entry;

} hf_mem_layout;

#ifdef __cplusplus
}
#endif

#endif /* DA6883A3_A70F_4690_A4FA_56644987725A */
