#ifndef UTILS_CUSPARSE_ERR_CUH
#define UTILS_CUSPARSE_ERR_CUH

/**
 * @file cuda_err.cuh
 * @author Jiannan Tian
 * @brief CUDA runtime error handling macros.
 * @version 0.2
 * @date 2020-09-20
 * Created on: 2019-10-08
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>

// block cusparse for generic testing
#ifndef noCUSPARSE

static void check_cusparse_error(cusparseStatus_t status, const char* file, int line)
{
    if (CUSPARSE_STATUS_SUCCESS != status) {
        printf("\nCUSPARSE status reference (as of CUDA 11):\n");
        printf("CUSPARSE_STATUS_SUCCESS                   -> %d\n", CUSPARSE_STATUS_SUCCESS);
        printf("CUSPARSE_STATUS_NOT_INITIALIZED           -> %d\n", CUSPARSE_STATUS_NOT_INITIALIZED);
        printf("CUSPARSE_STATUS_ALLOC_FAILED              -> %d\n", CUSPARSE_STATUS_ALLOC_FAILED);
        printf("CUSPARSE_STATUS_INVALID_VALUE             -> %d\n", CUSPARSE_STATUS_INVALID_VALUE);
        printf("CUSPARSE_STATUS_ARCH_MISMATCH             -> %d\n", CUSPARSE_STATUS_ARCH_MISMATCH);
        printf("CUSPARSE_STATUS_EXECUTION_FAILED          -> %d\n", CUSPARSE_STATUS_EXECUTION_FAILED);
        printf("CUSPARSE_STATUS_INTERNAL_ERROR            -> %d\n", CUSPARSE_STATUS_INTERNAL_ERROR);
        printf("CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED -> %d\n", CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
#if (CUDART_VERSION == 1010)
        printf("CUSPARSE_STATUS_NOT_SUPPORTED             -> %d\n", CUSPARSE_STATUS_NOT_SUPPORTED);
#endif
#if (CUDART_VERSION == 1100)
        printf("CUSPARSE_STATUS_INSUFFICIENT_RESOURCES    -> %d\n", CUSPARSE_STATUS_INSUFFICIENT_RESOURCES);
#endif
#if (CUDART_VERSION == 1100)
        printf("CUSPARSE_STATUS_INSUFFICIENT_RESOURCES    -> %d\n", CUSPARSE_STATUS_INSUFFICIENT_RESOURCES);
#endif
        printf("\n");

#if (CUDART_VERSION >= 1010)
        printf(
            "CUSPARSE API failed at \e[31m\e[1m%s:%d\e[0m with error: %s (%d)\n", file, line,
            cusparseGetErrorString(status), status);
#endif
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUSPARSE(err) (check_cusparse_error(err, __FILE__, __LINE__))

#endif

#endif
