/**
 * @file rt_config.h
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-06-03
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef BE11A946_C31B_4424_94ED_36C32DE6D824
#define BE11A946_C31B_4424_94ED_36C32DE6D824

#ifdef __cplusplus
extern "C" {
#endif

#include "cusz/type.h"
#include "mem/layout.h"

typedef psz_error_status psz_error_status;

// TODO move to some type.h
typedef struct psz_device_property {
    int sm_count, max_blockdim;
    int max_shmem_bytes, max_shmem_bytes_opt_in;
} psz_device_property;

psz_error_status psz_query_device(psz_device_property* prop);

psz_error_status psz_launch_p2013Histogram(
    psz_device_property* prop,
    uint32_t*            in_data,
    size_t const         in_len,
    uint32_t*            out_freq,
    int const            num_buckets,
    float*               milliseconds,
    void*                stream);

psz_error_status psz_hf_tune_coarse_encoding(size_t const len, psz_device_property* prop, int* sublen, int* pardeg);
int              psz_hf_revbook_nbyte(int booklen, int symbol_bytewidth);
size_t           paz_hf_max_compressed_bytes(size_t datalen);

#ifdef __cplusplus
}
#endif

#endif /* BE11A946_C31B_4424_94ED_36C32DE6D824 */
