/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>          // cusparseGather
#include <stdio.h>             // printf
#include <stdlib.h>            // EXIT_FAILURE
#include <iostream>

#include "../src/utils/cuda_err.cuh"

using namespace std;

//#define CHECK_CUDA(func)                                                                                              \
//    {                                                                                                                 \
//        cudaError_t status = (func);                                                                                  \
//        if (status != cudaSuccess) {                                                                                  \
//            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status); \
//            return EXIT_FAILURE;                                                                                      \
//        }                                                                                                             \
//    }
//
//#define CHECK_CUSPARSE(func)                                                                                      \
//    {                                                                                                             \
//        cusparseStatus_t status = (func);                                                                         \
//        if (status != CUSPARSE_STATUS_SUCCESS) {                                                                  \
//            printf(                                                                                               \
//                "CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), \
//                status);                                                                                          \
//            return EXIT_FAILURE;                                                                                  \
//        }                                                                                                         \
//    }

void run(int size, int step, int nnz)
{
    int*   hX_indices = new int[nnz]();
    float* hX_values  = new float[nnz]();
    float* hX_result  = new float[nnz]();
    float* hY         = new float[size]();

    {
        auto i = 0;
        while (i < nnz) {
            hX_indices[i] = i * step;
            hY[i * step]  = i;
            hX_result[i]  = i;
            i++;
        }
    }

    //    int   hX_indices[] = {0, 3, 4, 7};
    //    float hX_values[]  = {0.0f, 0.0f, 0.0f, 0.0f};
    //    float hY[]         = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    //    float hX_result[]  = {1.0f, 4.0f, 5.0f, 8.0f};
    //--------------------------------------------------------------------------
    // Device memory management
    int*   dX_indices;
    float *dY, *dX_values;
    CHECK_CUDA(cudaMalloc((void**)&dX_indices, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dX_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dY, size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dX_indices, hX_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dX_values, hX_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dY, hY, size * sizeof(float), cudaMemcpyHostToDevice));
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    // Create sparse vector X
    CHECK_CUSPARSE(cusparseCreateSpVec(
        &vecX, size, nnz, dX_indices, dX_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // Create dense vector y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, size, dY, CUDA_R_32F));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // execute Axpby
    CHECK_CUSPARSE(cusparseGather(handle, vecY, vecX));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << milliseconds << ',';

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA(cudaMemcpy(hX_values, dX_values, nnz * sizeof(float), cudaMemcpyDeviceToHost));
    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if (hX_values[i] != hX_result[i]) {  // direct floating point comparison
            correct = 0;                     // is not reliable in standard code
            break;
        }
    }
    if (!correct) printf("gather_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dX_indices));
    CHECK_CUDA(cudaFree(dX_values));
    CHECK_CUDA(cudaFree(dY));

    delete[] hX_indices;
    delete[] hX_values;
    delete[] hX_result;
    delete[] hY;
}

int main()
{
    // Host problem definition
    int   size_list[]     = {16, 32, 64, 128, 256, 512, 1024};
    float sparsity_list[] = {0.05, 0.02, 1e-2, 5e-3, 2e-3, 1e-3};

    for (auto& size_MiB : size_list) {
        cout << size_MiB << ',';
        for (auto& sparsity : sparsity_list) {
            auto size = size_MiB * 1024 * 1024;
            int  step = size * sparsity;
            int  nnz  = size / step;
            run(size, step, nnz);
        }
        cout << endl;
    }
    return EXIT_SUCCESS;
}
