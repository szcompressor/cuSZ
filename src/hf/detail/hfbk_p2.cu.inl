/**
 * @file hfbk_p2.cu.inl
 * @author Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Parallel Huffman Construction to generates canonical forward codebook.
 *        Based on [Ostadzadeh et al. 2007] (https://dblp.org/rec/conf/pdpta/OstadzadehEZMB07.bib)
 *        "A Two-phase Practical Parallel Algorithm for Construction of Huffman Codes".
 * @version 0.1
 * @date 2020-10-24
 * (created) 2020-05 (rev) 2021-06-21
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef A76584FA_A629_4AF8_930B_9B1FB56213C8
#define A76584FA_A629_4AF8_930B_9B1FB56213C8

#include <cooperative_groups.h>
#include <cuda.h>

#include "busyheader.hh"
#include "hf/hf_bookg.hh"
#include "typing.hh"
#include "utils/config.hh"
#include "utils/err.hh"
#include "utils/format.hh"
#include "utils/io.hh"
#include "utils/timer.hh"

namespace hf_detail {
template <typename T>
__global__ void GPU_FillArraySequence(T* array, unsigned int size)
{
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread < size) { array[thread] = thread; }
}

template <typename T>
__global__ void GPU_GetFirstNonzeroIndex(T* array, unsigned int size, unsigned int* result)
{
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (array[thread] != 0) { atomicMin(result, thread); }
}

template <typename H, typename T>
__global__ void GPU_ReorderByIndex(H* array, T* index, unsigned int size)
{
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    H            temp;
    T            newIndex;
    if (thread < size) {
        temp                 = array[thread];
        newIndex             = index[thread];
        array[(int)newIndex] = temp;
    }
}

template <typename T>
__global__ void GPU_ReverseArray(T* array, unsigned int size)
{
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread < size / 2) {
        T temp                   = array[thread];
        array[thread]            = array[size - thread - 1];
        array[size - thread - 1] = temp;
    }
}

__global__ void GPU_GetMaxCWLength(unsigned int* CL, unsigned int size, unsigned int* result)
{
    (void)size;
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread == 0) { *result = CL[0]; }
}

}  // namespace hf_detail

// Parallel codebook generation wrapper
template <typename T, typename H>
void psz::hf_buildbook_cu(
    uint32_t*    freq,
    int const    dict_size,
    H*           codebook,
    uint8_t*     revbook,
    int const    revbook_nbyte,
    float*       _time_book,
    void*        stream)
{
    // Metadata
    auto type_bw  = sizeof(H) * 8;
    auto _d_first = reinterpret_cast<H*>(revbook);
    auto _d_entry = reinterpret_cast<H*>(revbook + (sizeof(H) * type_bw));
    auto _d_qcode = reinterpret_cast<T*>(revbook + (sizeof(H) * 2 * type_bw));

    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING(stream);

    // Sort Qcodes by frequency
    int nblocks = (dict_size / 1024) + 1;
    hf_detail::GPU_FillArraySequence<T><<<nblocks, 1024>>>(_d_qcode, (unsigned int)dict_size);
    cudaStreamSynchronize((cudaStream_t)stream);

    lambda_sort_by_freq(freq, dict_size, _d_qcode);
    cudaStreamSynchronize((cudaStream_t)stream);

    unsigned int* d_first_nonzero_index;
    unsigned int  first_nonzero_index = dict_size;
    cudaMalloc(&d_first_nonzero_index, sizeof(unsigned int));
    cudaMemcpy(d_first_nonzero_index, &first_nonzero_index, sizeof(unsigned int), cudaMemcpyHostToDevice);
    hf_detail::GPU_GetFirstNonzeroIndex<unsigned int><<<nblocks, 1024>>>(freq, dict_size, d_first_nonzero_index);
    cudaStreamSynchronize((cudaStream_t)stream);
    cudaMemcpy(&first_nonzero_index, d_first_nonzero_index, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_first_nonzero_index);

    int           nz_dict_size   = dict_size - first_nonzero_index;
    unsigned int* _nz_d_freq     = freq + first_nonzero_index;
    H*            _nz_d_codebook = codebook + first_nonzero_index;
    int           nz_nblocks     = (nz_dict_size / 1024) + 1;

    // Memory Allocation -- Perhaps put in another wrapper
    // clang-format off
    unsigned int *CL         = nullptr;
    /*unsigned int* lNodesFreq*/         int *lNodesLeader = nullptr;
    unsigned int *iNodesFreq = nullptr;  int *iNodesLeader = nullptr;
    unsigned int *tempFreq   = nullptr;  int *tempIsLeaf   = nullptr;  int *tempIndex = nullptr;
    unsigned int *copyFreq   = nullptr;  int *copyIsLeaf   = nullptr;  int *copyIndex = nullptr;
    cudaMalloc(&CL,           nz_dict_size * sizeof(unsigned int) );
    cudaMalloc(&lNodesLeader, nz_dict_size * sizeof(int)          );
    cudaMalloc(&iNodesFreq,   nz_dict_size * sizeof(unsigned int) );
    cudaMalloc(&iNodesLeader, nz_dict_size * sizeof(int)          );
    cudaMalloc(&tempFreq,     nz_dict_size * sizeof(unsigned int) );
    cudaMalloc(&tempIsLeaf,   nz_dict_size * sizeof(int)          );
    cudaMalloc(&tempIndex,    nz_dict_size * sizeof(int)          );
    cudaMalloc(&copyFreq,     nz_dict_size * sizeof(unsigned int) );
    cudaMalloc(&copyIsLeaf,   nz_dict_size * sizeof(int)          );
    cudaMalloc(&copyIndex,    nz_dict_size * sizeof(int)          );
    cudaMemset(CL, 0,         nz_dict_size * sizeof(int)          );
    // clang-format on

    // Grid configuration for CL -- based on Cooperative Groups
    int            cg_mblocks;
    int            cg_blocks_sm;
    int            device_id;
    int            mthreads = 32;  // 1 warp
    cudaDeviceProp deviceProp;
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&deviceProp, device_id);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &cg_blocks_sm, par_huffman::GPU_GenerateCL<unsigned int>, mthreads, 5 * sizeof(int32_t) + 32 * sizeof(int32_t));
    cg_mblocks = deviceProp.multiProcessorCount * cg_blocks_sm;

    int ELTS_PER_SEQ_MERGE = 16;
    int mblocks            = std::min(cg_mblocks, (nz_dict_size / ELTS_PER_SEQ_MERGE) + 1);

    // Exit if not enough exposed parallelism -- TODO modify kernels so this is unneeded
    int tthreads = mthreads * mblocks;
    if (tthreads < nz_dict_size) {
        cout << LOG_ERR << "Insufficient on-device parallelism to construct a " << nz_dict_size
             << " non-zero item codebook" << endl;
        cout << LOG_ERR << "Provided parallelism: " << mblocks << " blocks, " << mthreads << " threads, " << tthreads
             << " total" << endl
             << endl;
        // cout << LOG_ERR << "Exiting cuSZ ..." << endl;
        throw std::system_error();
        // exit(1);
    }

    uint32_t* diagonal_path_intersections;
    cudaMalloc(&diagonal_path_intersections, (2 * (mblocks + 1)) * sizeof(uint32_t));

    // Codebook already init'ed
    cudaStreamSynchronize((cudaStream_t)stream);

    // Call first kernel
    // Collect arguments
    void* CL_Args[] = {(void*)&_nz_d_freq,   (void*)&CL,
                       (void*)&nz_dict_size, (void*)&_nz_d_freq,
                       (void*)&lNodesLeader, (void*)&iNodesFreq,
                       (void*)&iNodesLeader, (void*)&tempFreq,
                       (void*)&tempIsLeaf,   (void*)&tempIndex,
                       (void*)&copyFreq,     (void*)&copyIsLeaf,
                       (void*)&copyIndex,    (void*)&diagonal_path_intersections,
                       (void*)&mblocks,      (void*)&mthreads};
    // Cooperative Launch
    cudaLaunchCooperativeKernel(
        (void*)par_huffman::GPU_GenerateCL<unsigned int>, mblocks, mthreads, CL_Args,
        5 * sizeof(int32_t) + 32 * sizeof(int32_t));
    cudaStreamSynchronize((cudaStream_t)stream);

    // Exits if the highest codeword length is greater than what
    // the adaptive representation can handle
    // TODO do  proper cleanup

    unsigned int* d_max_CL;
    unsigned int  max_CL;
    cudaMalloc(&d_max_CL, sizeof(unsigned int));
    hf_detail::GPU_GetMaxCWLength<<<1, 1>>>(CL, nz_dict_size, d_max_CL);
    cudaStreamSynchronize((cudaStream_t)stream);
    cudaMemcpy(&max_CL, d_max_CL, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_max_CL);

    int max_CW_bits = (sizeof(H) * 8) - 8;
    if (max_CL > max_CW_bits) {
        cout << LOG_ERR << "Cannot store all Huffman codewords in " << max_CW_bits + 8 << "-bit representation" << endl;
        cout << LOG_ERR << "Huffman codeword representation requires at least " << max_CL + 8
             << " bits (longest codeword: " << max_CL << " bits)" << endl;
        // cout << LOG_ERR << "(Consider running with -H 8 for 8-byte representation)" << endl << endl;
        // cout << LOG_ERR << "Exiting cuSZ ..." << endl;
        // exit(1);
        throw std::runtime_error("Falling back to 8-byte Codec.");
    }

    // Configure CW for 1024 threads/block
    int cg_cw_mblocks = (cg_mblocks * mthreads) / 1024;
    int cw_mblocks    = std::min(cg_cw_mblocks, nz_nblocks);

    // Exit if not enough exposed parallelism -- TODO modify kernels so this is unneeded
    int cw_tthreads = cw_mblocks * 1024;
    if (cw_tthreads < nz_dict_size) {
        cout << LOG_ERR << "Insufficient on-device parallelism to construct a " << nz_dict_size
             << " non-zero item codebook" << endl;
        cout << LOG_ERR << "Provided parallelism: " << cw_mblocks << " blocks, " << 1024 << " threads, " << cw_tthreads
             << " total" << endl
             << endl;
        // cout << LOG_ERR << "Exiting cuSZ ..." << endl;
        // exit(1);
        throw std::system_error();
    }

    void* CW_Args[] = {
        (void*)&CL,              //
        (void*)&_nz_d_codebook,  //
        (void*)&_d_first,        //
        (void*)&_d_entry,        //
        (void*)&nz_dict_size};

    // Call second kernel
    cudaLaunchCooperativeKernel(
        (void*)par_huffman::GPU_GenerateCW<unsigned int, H>,  //
        cw_mblocks,                                           //
        1024,                                                 //
        CW_Args);
    cudaStreamSynchronize((cudaStream_t)stream);

#ifdef D_DEBUG_PRINT
    print_codebook<H><<<1, 32>>>(codebook, dict_size);  // PASS
    cudaStreamSynchronize((cudaStream_t)stream);
#endif

    // Reverse _d_qcode and codebook
    hf_detail::GPU_ReverseArray<H><<<nblocks, 1024>>>(codebook, (unsigned int)dict_size);
    hf_detail::GPU_ReverseArray<T><<<nblocks, 1024>>>(_d_qcode, (unsigned int)dict_size);
    cudaStreamSynchronize((cudaStream_t)stream);

    hf_detail::GPU_ReorderByIndex<H, T><<<nblocks, 1024>>>(codebook, _d_qcode, (unsigned int)dict_size);
    cudaStreamSynchronize((cudaStream_t)stream);

    STOP_GPUEVENT_RECORDING((cudaStream_t)stream);
    TIME_ELAPSED_GPUEVENT(_time_book);
    DESTROY_GPUEVENT_PAIR;

    // Cleanup
    cudaFree(CL);
    cudaFree(lNodesLeader);
    cudaFree(iNodesFreq);
    cudaFree(iNodesLeader);
    cudaFree(tempFreq);
    cudaFree(tempIsLeaf);
    cudaFree(tempIndex);
    cudaFree(copyFreq);
    cudaFree(copyIsLeaf);
    cudaFree(copyIndex);
    cudaFree(diagonal_path_intersections);
    cudaStreamSynchronize((cudaStream_t)stream);

#ifdef D_DEBUG_PRINT
    print_codebook<H><<<1, 32>>>(codebook, dict_size);  // PASS
    cudaStreamSynchronize((cudaStream_t)stream);
#endif
}

#endif /* A76584FA_A629_4AF8_930B_9B1FB56213C8 */
