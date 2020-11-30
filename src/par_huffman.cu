/**
 * @file par_huffman.cu
 * @author Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Parallel Huffman Construction to generates canonical forward codebook.
 *        Based on [Ostadzadeh et al. 2007] (https://dblp.org/rec/conf/pdpta/OstadzadehEZMB07.bib)
 *        "A Two-phase Practical Parallel Algorithm for Construction of Huffman Codes".
 * @version 0.1
 * @date 2020-10-24
 * Created on: 2020-05
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "cuda_error_handling.cuh"
#include "cuda_mem.cuh"
#include "dbg_gpu_printing.cuh"
#include "format.hh"
#include "par_huffman.cuh"
#include "par_merge.cuh"

// Mathematically correct mod
#define MOD(a, b) ((((a) % (b)) + (b)) % (b))

// Parallel huffman code generation
// clang-format off
template <typename F>
__global__ void parHuff::GPU_GenerateCL(
    F*  histogram,  F* CL,  int size,
    /* Global Arrays */
    F* lNodesFreq,  int* lNodesLeader,
    F* iNodesFreq,  int* iNodesLeader,
    F* tempFreq,    int* tempIsLeaf,    int* tempIndex,
    F* copyFreq,    int* copyIsLeaf,    int* copyIndex,
    uint32_t* diagonal_path_intersections, int mblocks, int mthreads)
{
    // clang-format on

    extern __shared__ int32_t shmem[];
    // Shared variables
    int32_t& x_top     = shmem[0];
    int32_t& y_top     = shmem[1];
    int32_t& x_bottom  = shmem[2];
    int32_t& y_bottom  = shmem[3];
    int32_t& found     = shmem[4];
    int32_t* oneorzero = &shmem[5];

    unsigned int       thread       = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int i            = thread;  // Adaptation for easier porting
    auto               current_grid = this_grid();

    /* Initialization */
    if (thread < size) {
        lNodesLeader[i] = -1;
        CL[i]           = 0;
    }

    if (thread == 0) {
        iNodesFront = 0;
        iNodesRear  = 0;
        lNodesCur   = 0;

        iNodesSize = 0;
    }
    current_grid.sync();

    /* While there is not exactly one internal node */
    while (lNodesCur < size || iNodesSize > 1) {
        /* Combine two most frequent nodes on same level */
        if (thread == 0) {
            F   midFreq[4];
            int midIsLeaf[4];
            for (int i = 0; i < 4; ++i) midFreq[i] = UINT_MAX;

            if (lNodesCur < size) {
                midFreq[0]   = lNodesFreq[lNodesCur];
                midIsLeaf[0] = 1;
            }
            if (lNodesCur < size - 1) {
                midFreq[1]   = lNodesFreq[lNodesCur + 1];
                midIsLeaf[1] = 1;
            }
            if (iNodesSize >= 1) {
                midFreq[2]   = iNodesFreq[iNodesFront];
                midIsLeaf[2] = 0;
            }
            if (iNodesSize >= 2) {
                midFreq[3]   = iNodesFreq[MOD(iNodesFront + 1, size)];
                midIsLeaf[3] = 0;
            }

            /* Select the minimum of minimums - 4elt sorting network */
            /* TODO There's likely a good 1-warp faster way to do this */
            {
                F   tempFreq;
                int tempIsLeaf;
                if (midFreq[1] > midFreq[3]) {
                    tempFreq     = midFreq[1];
                    midFreq[1]   = midFreq[3];
                    midFreq[3]   = tempFreq;
                    tempIsLeaf   = midIsLeaf[1];
                    midIsLeaf[1] = midIsLeaf[3];
                    midIsLeaf[3] = tempIsLeaf;
                }
                if (midFreq[0] > midFreq[2]) {
                    tempFreq     = midFreq[0];
                    midFreq[0]   = midFreq[2];
                    midFreq[2]   = tempFreq;
                    tempIsLeaf   = midIsLeaf[0];
                    midIsLeaf[0] = midIsLeaf[2];
                    midIsLeaf[2] = tempIsLeaf;
                }
                if (midFreq[0] > midFreq[1]) {
                    tempFreq     = midFreq[0];
                    midFreq[0]   = midFreq[1];
                    midFreq[1]   = tempFreq;
                    tempIsLeaf   = midIsLeaf[0];
                    midIsLeaf[0] = midIsLeaf[1];
                    midIsLeaf[1] = tempIsLeaf;
                }
                if (midFreq[2] > midFreq[3]) {
                    tempFreq     = midFreq[2];
                    midFreq[2]   = midFreq[3];
                    midFreq[3]   = tempFreq;
                    tempIsLeaf   = midIsLeaf[2];
                    midIsLeaf[2] = midIsLeaf[3];
                    midIsLeaf[3] = tempIsLeaf;
                }
                if (midFreq[1] > midFreq[2]) {
                    tempFreq     = midFreq[1];
                    midFreq[1]   = midFreq[2];
                    midFreq[2]   = tempFreq;
                    tempIsLeaf   = midIsLeaf[1];
                    midIsLeaf[1] = midIsLeaf[2];
                    midIsLeaf[2] = tempIsLeaf;
                }
            }

            minFreq = midFreq[0];
            if (midFreq[1] < UINT_MAX) { minFreq += midFreq[1]; }
            iNodesFreq[iNodesRear]   = minFreq;
            iNodesLeader[iNodesRear] = -1;

            /* If is leaf */
            if (midIsLeaf[0]) {
                lNodesLeader[lNodesCur] = iNodesRear;
                ++CL[lNodesCur], ++lNodesCur;
            }
            else {
                iNodesLeader[iNodesFront] = iNodesRear;
                iNodesFront               = MOD(iNodesFront + 1, size);
            }
            if (midIsLeaf[1]) {
                lNodesLeader[lNodesCur] = iNodesRear;
                ++CL[lNodesCur], ++lNodesCur;
            }
            else {
                iNodesLeader[iNodesFront] = iNodesRear;
                iNodesFront               = MOD(iNodesFront + 1, size); /* ? */
            }

            // iNodesRear = MOD(iNodesRear + 1, size);

            iNodesSize = MOD(iNodesRear - iNodesFront, size);
        }

        // int curLeavesNum;
        /* Select elements to copy -- parallelized */
        curLeavesNum = 0;
        current_grid.sync();
        if (i >= lNodesCur && i < size) {
            // Parallel component
            int threadCurLeavesNum;
            if (lNodesFreq[i] <= minFreq) {
                threadCurLeavesNum = i - lNodesCur + 1;
                // Atomic max -- Largest valid index
                atomicMax(&curLeavesNum, threadCurLeavesNum);
            }

            if (i - lNodesCur < curLeavesNum) {
                copyFreq[i - lNodesCur]   = lNodesFreq[i];
                copyIndex[i - lNodesCur]  = i;
                copyIsLeaf[i - lNodesCur] = 1;
            }
        }

        current_grid.sync();

        /* Updates Iterators */
        if (thread == 0) {
            mergeRear  = iNodesRear;
            mergeFront = iNodesFront;

            if ((curLeavesNum + iNodesSize) % 2 == 0) { iNodesFront = iNodesRear; }
            /* Odd number of nodes to merge - leave out one*/
            else if (
                (iNodesSize != 0)                                                                        //
                and (curLeavesNum == 0                                                                   //
                     or (histogram[lNodesCur + curLeavesNum] <= iNodesFreq[MOD(iNodesRear - 1, size)]))  //
            ) {
                mergeRear   = MOD(mergeRear - 1, size);
                iNodesFront = MOD(iNodesRear - 1, size);
            }
            else {
                iNodesFront = iNodesRear;
                --curLeavesNum;
            }

            lNodesCur  = lNodesCur + curLeavesNum;
            iNodesRear = MOD(iNodesRear + 1, size);
        }
        current_grid.sync();

        /* Parallelized Merging Phase */

        /*if (thread == 0) {
            merge(copyFreq, copyIndex, copyIsLeaf, 0, curLeavesNum,
                    iNodesFreq, mergeFront, mergeRear, size,
                    tempFreq, tempIndex, tempIsLeaf, tempLength);
                    }*/

        parMerge(
            copyFreq, copyIndex, copyIsLeaf, 0, curLeavesNum,  //
            iNodesFreq, mergeFront, mergeRear, size,           //
            tempFreq, tempIndex, tempIsLeaf, tempLength,       //
            diagonal_path_intersections, mblocks, mthreads,    //
            x_top, y_top, x_bottom, y_bottom, found, oneorzero);
        current_grid.sync();

        /* Melding phase -- New */
        if (thread < tempLength / 2) {
            int ind           = MOD(iNodesRear + i, size);
            iNodesFreq[ind]   = tempFreq[(2 * i)] + tempFreq[(2 * i) + 1];
            iNodesLeader[ind] = -1;

            if (tempIsLeaf[(2 * i)]) {
                lNodesLeader[tempIndex[(2 * i)]] = ind;
                ++CL[tempIndex[(2 * i)]];
            }
            else {
                iNodesLeader[tempIndex[(2 * i)]] = ind;
            }
            if (tempIsLeaf[(2 * i) + 1]) {
                lNodesLeader[tempIndex[(2 * i) + 1]] = ind;
                ++CL[tempIndex[(2 * i) + 1]];
            }
            else {
                iNodesLeader[tempIndex[(2 * i) + 1]] = ind;
            }
        }
        current_grid.sync();

        if (thread == 0) { iNodesRear = MOD(iNodesRear + (tempLength / 2), size); }
        current_grid.sync();

        /* Update leaders */
        if (thread < size) {
            if (lNodesLeader[i] != -1) {
                if (iNodesLeader[lNodesLeader[i]] != -1) {
                    lNodesLeader[i] = iNodesLeader[lNodesLeader[i]];
                    ++CL[i];
                }
            }
        }
        current_grid.sync();

        if (thread == 0) { iNodesSize = MOD(iNodesRear - iNodesFront, size); }
        current_grid.sync();
    }
}

// Parallelized with atomic writes, but could replace with Jiannan's similar code
template <typename F, typename H>
__global__ void parHuff::GPU_GenerateCW(F* CL, H* CW, H* first, H* entry, int size)
{
    unsigned int       thread       = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int i            = thread;  // Porting convenience
    auto               current_grid = this_grid();
    auto               type_bw      = sizeof(H) * 8;

    /* Reverse in place - Probably a more CUDA-appropriate way */
    if (thread < size / 2) {
        F temp           = CL[i];
        CL[i]            = CL[size - i - 1];
        CL[size - i - 1] = temp;
    }
    current_grid.sync();

    if (thread == 0) {
        CCL        = CL[0];
        CDPI       = 0;
        newCDPI    = size - 1;
        entry[CCL] = 0;

        // Edge case -- only one input symbol
        CW[CDPI]       = 0;
        first[CCL]     = CW[CDPI] ^ (((H)1 << (H)CL[CDPI]) - 1);
        entry[CCL + 1] = 1;
    }
    current_grid.sync();

    // Initialize first and entry arrays
    if (thread < CCL) {
        // Initialization of first to Max ensures that unused code
        // lengths are skipped over in decoding.
        first[i] = std::numeric_limits<H>::max();
        entry[i] = 0;
    }
    // Initialize first element of entry
    current_grid.sync();

    while (CDPI < size - 1) {
        // CDPI update
        if (i < size - 1 && CL[i + 1] > CCL) { atomicMin(&newCDPI, i); }
        current_grid.sync();

        // Last element to update
        const int updateEnd = (newCDPI >= size - 1) ? type_bw : CL[newCDPI + 1];
        // Fill base
        const int curEntryVal = entry[CCL];
        // Number of elements of length CCL
        const int numCCL = (newCDPI - CDPI + 1);

        // Get first codeword
        if (i == 0) {
            if (CDPI == 0) { CW[newCDPI] = 0; }
            else {
                CW[newCDPI] = CW[CDPI];  // Pre-stored
            }
        }
        current_grid.sync();

        if (i < size) {
            // Parallel canonical codeword generation
            if (i >= CDPI && i < newCDPI) { CW[i] = CW[newCDPI] + (newCDPI - i); }
        }

        // Update entry and first arrays in O(1) time
        if (thread > CCL && thread < updateEnd) { entry[i] = curEntryVal + numCCL; }
        // Add number of entries to next CCL
        if (thread == 0) {
            if (updateEnd < type_bw) { entry[updateEnd] = curEntryVal + numCCL; }
        }
        current_grid.sync();

        // Update first array in O(1) time
        if (thread == CCL) {
            // Flip least significant CL[CDPI] bits
            first[CCL] = CW[CDPI] ^ (((H)1 << (H)CL[CDPI]) - 1);
        }
        if (thread > CCL && thread < updateEnd) { first[i] = std::numeric_limits<H>::max(); }
        current_grid.sync();

        if (thread == 0) {
            if (newCDPI < size - 1) {
                int CLDiff = CL[newCDPI + 1] - CL[newCDPI];
                // Add and shift -- Next canonical code
                CW[newCDPI + 1] = ((CW[CDPI] + 1) << CLDiff);
                CCL             = CL[newCDPI + 1];

                ++newCDPI;
            }

            // Update CDPI to newCDPI after codeword length increase
            CDPI    = newCDPI;
            newCDPI = size - 1;
        }
        current_grid.sync();
    }

    if (thread < size) {
        /* Make encoded codeword compatible with CUSZ */
        CW[i] = (CW[i] | (((H)CL[i] & (H)0xffu) << ((sizeof(H) * 8) - 8))) ^ (((H)1 << (H)CL[i]) - 1);
    }
    current_grid.sync();

    /* Reverse partial codebook */
    if (thread < size / 2) {
        H temp           = CW[i];
        CW[i]            = CW[size - i - 1];
        CW[size - i - 1] = temp;
    }
}

// Helper implementations
template <typename T>
__global__ void GPU_FillArraySequence(T* array, unsigned int size)
{
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread < size) { array[thread] = thread; }
}

// Precondition -- Result is preset to be equal to size
template <typename T>
__global__ void GPU_GetFirstNonzeroIndex(T* array, unsigned int size, unsigned int* result)
{
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (array[thread] != 0) { atomicMin(result, thread); }
}

__global__ void GPU_GetMaxCWLength(unsigned int* CL, unsigned int size, unsigned int* result)
{
    (void)size;
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread == 0) { *result = CL[0]; }
}

// Reorders given a set of indices. Programmer must ensure that all index[i]
// are unique or else race conditions may occur
template <typename T, typename Q>
__global__ void GPU_ReorderByIndex(T* array, Q* index, unsigned int size)
{
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    T            temp;
    Q            newIndex;
    if (thread < size) {
        temp            = array[thread];
        newIndex        = index[thread];
        array[newIndex] = temp;
    }
}

// Reverses a given array.
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

// Parallel codebook generation wrapper
template <typename Q, typename H>
void ParGetCodebook(int dict_size, unsigned int* _d_freq, H* _d_codebook, uint8_t* _d_decode_meta)
{
    // Metadata
    auto type_bw  = sizeof(H) * 8;
    auto _d_first = reinterpret_cast<H*>(_d_decode_meta);
    auto _d_entry = reinterpret_cast<H*>(_d_decode_meta + (sizeof(H) * type_bw));
    auto _d_qcode = reinterpret_cast<Q*>(_d_decode_meta + (sizeof(H) * 2 * type_bw));

    // Sort Qcodes by frequency
    int nblocks = (dict_size / 1024) + 1;
    GPU_FillArraySequence<Q><<<nblocks, 1024>>>(_d_qcode, (unsigned int)dict_size);
    cudaDeviceSynchronize();

    SortByFreq(_d_freq, _d_qcode, dict_size);
    cudaDeviceSynchronize();

    unsigned int* d_first_nonzero_index;
    unsigned int  first_nonzero_index = dict_size;
    cudaMalloc(&d_first_nonzero_index, sizeof(unsigned int));
    cudaMemcpy(d_first_nonzero_index, &first_nonzero_index, sizeof(unsigned int), cudaMemcpyHostToDevice);
    GPU_GetFirstNonzeroIndex<unsigned int><<<nblocks, 1024>>>(_d_freq, dict_size, d_first_nonzero_index);
    cudaDeviceSynchronize();
    cudaMemcpy(&first_nonzero_index, d_first_nonzero_index, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_first_nonzero_index);

    int           nz_dict_size   = dict_size - first_nonzero_index;
    unsigned int* _nz_d_freq     = _d_freq + first_nonzero_index;
    H*            _nz_d_codebook = _d_codebook + first_nonzero_index;
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
        &cg_blocks_sm, parHuff::GPU_GenerateCL<unsigned int>, mthreads, 5 * sizeof(int32_t) + 32 * sizeof(int32_t));
    cg_mblocks = deviceProp.multiProcessorCount * cg_blocks_sm;

    int ELTS_PER_SEQ_MERGE = 16;
    int mblocks            = std::min(cg_mblocks, (nz_dict_size / ELTS_PER_SEQ_MERGE) + 1);

    // Exit if not enough exposed parallelism -- TODO modify kernels so this is unneeded
    int tthreads = mthreads * mblocks;
    if (tthreads < nz_dict_size) {
        cout << log_err << "Insufficient on-device parallelism to construct a " << nz_dict_size
             << " non-zero item codebook" << endl;
        cout << log_err << "Provided parallelism: " << mblocks << " blocks, " << mthreads << " threads, " << tthreads
             << " total" << endl
             << endl;
        cout << log_err << "Exiting cuSZ ..." << endl;
        exit(1);
    }

    uint32_t* diagonal_path_intersections;
    cudaMalloc(&diagonal_path_intersections, (2 * (mblocks + 1)) * sizeof(uint32_t));

    // Codebook already init'ed
    cudaDeviceSynchronize();

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
        (void*)parHuff::GPU_GenerateCL<unsigned int>, mblocks, mthreads, CL_Args,
        5 * sizeof(int32_t) + 32 * sizeof(int32_t));
    cudaDeviceSynchronize();

    // Exits if the highest codeword length is greater than what
    // the adaptive representation can handle
    // TODO do  proper cleanup

    unsigned int* d_max_CL;
    unsigned int  max_CL;
    cudaMalloc(&d_max_CL, sizeof(unsigned int));
    GPU_GetMaxCWLength<<<1, 1>>>(CL, nz_dict_size, d_max_CL);
    cudaDeviceSynchronize();
    cudaMemcpy(&max_CL, d_max_CL, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_max_CL);

    int max_CW_bits = (sizeof(H) * 8) - 8;
    if (max_CL > max_CW_bits) {
        cout << log_err << "Cannot store all Huffman codewords in " << max_CW_bits + 8 << "-bit representation" << endl;
        cout << log_err << "Huffman codeword representation requires at least " << max_CL + 8
             << " bits (longest codeword: " << max_CL << " bits)" << endl;
        cout << log_err << "(Consider running with -H 64)" << endl << endl;
        cout << log_err << "Exiting cuSZ ..." << endl;
        exit(1);
    }

    // Configure CW for 1024 threads/block
    int cg_cw_mblocks = (cg_mblocks * mthreads) / 1024;
    int cw_mblocks    = std::min(cg_cw_mblocks, nz_nblocks);

    // Exit if not enough exposed parallelism -- TODO modify kernels so this is unneeded
    int cw_tthreads = cw_mblocks * 1024;
    if (cw_tthreads < nz_dict_size) {
        cout << log_err << "Insufficient on-device parallelism to construct a " << nz_dict_size
             << " non-zero item codebook" << endl;
        cout << log_err << "Provided parallelism: " << cw_mblocks << " blocks, " << 1024 << " threads, " << cw_tthreads
             << " total" << endl
             << endl;
        cout << log_err << "Exiting cuSZ ..." << endl;
        exit(1);
    }

    void* CW_Args[] = {
        (void*)&CL,              //
        (void*)&_nz_d_codebook,  //
        (void*)&_d_first,        //
        (void*)&_d_entry,        //
        (void*)&nz_dict_size};

    // Call second kernel
    cudaLaunchCooperativeKernel(
        (void*)parHuff::GPU_GenerateCW<unsigned int, H>,  //
        cw_mblocks,                                       //
        1024,                                             //
        CW_Args);
    cudaDeviceSynchronize();

#ifdef D_DEBUG_PRINT
    print_codebook<H><<<1, 32>>>(_d_codebook, dict_size);  // PASS
    cudaDeviceSynchronize();
#endif

    // Reverse _d_qcode and _d_codebook
    GPU_ReverseArray<H><<<nblocks, 1024>>>(_d_codebook, (unsigned int)dict_size);
    GPU_ReverseArray<Q><<<nblocks, 1024>>>(_d_qcode, (unsigned int)dict_size);
    cudaDeviceSynchronize();

    GPU_ReorderByIndex<H, Q><<<nblocks, 1024>>>(_d_codebook, _d_qcode, (unsigned int)dict_size);
    cudaDeviceSynchronize();

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
    cudaDeviceSynchronize();

#ifdef D_DEBUG_PRINT
    print_codebook<H><<<1, 32>>>(_d_codebook, dict_size);  // PASS
    cudaDeviceSynchronize();
#endif
}

// Specialize wrapper
template void ParGetCodebook<uint8_t, uint32_t>(int dict_size, unsigned int* freq, uint32_t* codebook, uint8_t* meta);
template void ParGetCodebook<uint8_t, uint64_t>(int dict_size, unsigned int* freq, uint64_t* codebook, uint8_t* meta);
template void ParGetCodebook<uint16_t, uint32_t>(int dict_size, unsigned int* freq, uint32_t* codebook, uint8_t* meta);
template void ParGetCodebook<uint16_t, uint64_t>(int dict_size, unsigned int* freq, uint64_t* codebook, uint8_t* meta);
template void ParGetCodebook<uint32_t, uint32_t>(int dict_size, unsigned int* freq, uint32_t* codebook, uint8_t* meta);
template void ParGetCodebook<uint32_t, uint64_t>(int dict_size, unsigned int* freq, uint64_t* codebook, uint8_t* meta);
