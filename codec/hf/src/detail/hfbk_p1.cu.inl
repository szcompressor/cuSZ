/**
 * @file hfbk_p1.cu.inl
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

#ifndef C883A574_4491_40E8_A083_1B6E8FB56670
#define C883A574_4491_40E8_A083_1B6E8FB56670

#include <cooperative_groups.h>
#include <cuda.h>
#include "busyheader.hh"
#include "hf/hf_bookg.hh"
#include "par_merge.inl"
#include "typing.hh"
#include "utils/config.hh"
#include "utils/err.hh"
#include "utils/format.hh"
#include "utils/io.hh"
#include "utils/timer.hh"

using std::cout;
using std::endl;
namespace cg = cooperative_groups;

// GenerateCL Locals
__device__ int iNodesFront = 0;
__device__ int iNodesRear  = 0;
__device__ int lNodesCur   = 0;

__device__ int iNodesSize = 0;
__device__ int curLeavesNum;

__device__ int minFreq;

__device__ int tempLength;

__device__ int mergeFront;
__device__ int mergeRear;

__device__ int lNodesIndex;

// GenerateCW Locals
__device__ int CCL;
__device__ int CDPI;
__device__ int newCDPI;

// Profiling
__device__ long long int s[10];
__device__ long long int st[10];

// Mathematically correct mod
#define MOD(a, b) ((((a) % (b)) + (b)) % (b))

// Parallel huffman code generation
// clang-format off
template <typename F>
__global__ void par_huffman::GPU_GenerateCL(
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
    auto               current_grid = cg::this_grid();

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
__global__ void par_huffman::GPU_GenerateCW(F* CL, H* CW, H* first, H* entry, int size)
{
    unsigned int       thread       = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int i            = thread;  // Porting convenience
    auto               current_grid = cg::this_grid();
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

#endif /* C883A574_4491_40E8_A083_1B6E8FB56670 */
