/*
 * Authors:
 *  Oded Green (ogreen@gatech.edu), Rob McColl (robert.c.mccoll@gmail.com)
 *  High Performance Computing Lab, Georgia Tech
 *
 * Future Publication:
 * GPU MergePath: A GPU Merging Algorithm
 * ACM International Conference on Supercomputing 2012
 * June 25-29 2012, San Servolo, Venice, Italy
 *
 * Copyright (c) 2012 Georgia Institute of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the Georgia Institute of Technology nor the names of
 *   its contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

//
// Modified by Cody Rivera, 6/2020
//

#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/merge.h>

#include <cooperative_groups.h>
#include "par_merge.cuh"

using namespace cooperative_groups;

#define MAX(X,Y) (((X) > (Y)) ? (X) : (Y))
#define MIN(X,Y) (((X) < (Y)) ? (X) : (Y))
// Mathematically correct modulo
#define MOD(a,b) ((((a)%(b))+(b))%(b))

// Profiling
__device__ long mergeProfile[2] = {0, 0};
__device__ long mergeProfileTotal[2] = {0, 0};

/* POSITIVEINFINITY
 * Returns maximum value of a type
 */
 template<typename vec_t>
 __host__ __device__ vec_t getPositiveInfinity() {
     vec_t tmp = 0;
     return positiveInfinity(tmp);
 }
 __host__ __device__ float positiveInfinity(float tmp) {
     return FLT_MAX;
 }
 __host__ __device__ double positiveInfinity(double tmp) {
     return DBL_MAX;
 }
 __host__ __device__ uint32_t positiveInfinity(uint32_t tmp) {
     return 0xFFFFFFFFUL;
 }
 __host__ __device__ uint64_t positiveInfinity(uint64_t tmp) {
     return 0xFFFFFFFFFFFFFFFFUL;
 }
 /* NEGATIVEINFINITY
  * Returns minimum value of a type
  */
 template<typename vec_t>
 __host__ __device__ vec_t getNegativeInfinity() {
     vec_t tmp = 0;
     return negativeInfinity(tmp);
 }
 __host__ __device__ float negativeInfinity(float tmp) {
     return FLT_MIN;
 }
 __host__ __device__ double negativeInfinity(double tmp) {
     return DBL_MIN;
 }
 __host__ __device__ uint32_t negativeInfinity(uint32_t tmp) {
     return 0;
 }
 __host__ __device__ uint64_t negativeInfinity(uint64_t tmp) {
     return 0;
 }
 
 /* RAND64
  * Gives up to 64-bits of pseudo-randomness
  * Note: not very "good" or "random"
  */
 template<typename vec_t>
 vec_t rand64() {
     vec_t rtn;
     do {
         uint32_t * rtn32 = (uint32_t *)&rtn;
         rtn32[0] = rand();
         if(sizeof(vec_t) > 4) rtn32[1] = rand();
     } while(!(rtn < getPositiveInfinity<vec_t>() &&
               rtn > getNegativeInfinity<vec_t>()));
     return rtn;
 }
 
 /* MERGETYPE
  * Performs <runs> merges of two sorted pseudorandom <vec_t> arrays of length <size>
  * Times the runs and reports on the average time
  * Checks the output of each merge for correctness
  */
 #define PADDING 1024
 template <typename F>
 __device__ void parMerge(F* copyFreq, int* copyIndex, int* copyIsLeaf, int cStart, int cEnd,
                          F* iNodesFreq, int iStart, int iEnd, int iNodesCap,
                          F* tempFreq, int* tempIndex, int* tempIsLeaf, int& tempLength,
                          uint32_t* diagonal_path_intersections, int blocks, int threads,
                          /* Shared Memory */
                          int32_t& x_top, int32_t& y_top, int32_t& x_bottom, int32_t& y_bottom,
                          int32_t& found, int32_t* oneorzero) 
{
     auto current_grid = this_grid();
     current_grid.sync();
     tempLength = (cEnd - cStart) + MOD(iEnd - iStart, iNodesCap);
 
     if (tempLength == 0) return;
 
     // Perform the global diagonal intersection serach to divide work among SMs
     // Dynamic parallelism -- make more efficient -- incorporate into configuration
     if (threadIdx.x == 0 && blockIdx.x == 0)
         mergeProfile[0] = clock64();
     cudaWorkloadDiagonals<F>
         (copyFreq, copyIndex, copyIsLeaf, cStart, cEnd,
          iNodesFreq, iStart, iEnd, iNodesCap,
          diagonal_path_intersections,
          x_top, y_top, x_bottom, y_bottom, found, oneorzero);
     current_grid.sync();
     if (threadIdx.x == 0 && blockIdx.x == 0)
         mergeProfileTotal[0] += clock64() - mergeProfile[0];
 
     // Merge between global diagonals independently on each block
     if (threadIdx.x == 0 && blockIdx.x == 0)
         mergeProfile[1] = clock64();
     cudaMergeSinglePath<F>
         (copyFreq, copyIndex, copyIsLeaf, cStart, cEnd,
          iNodesFreq, iStart, iEnd, iNodesCap,
          diagonal_path_intersections,
          tempFreq, tempIndex, tempIsLeaf, tempLength);
     current_grid.sync();
     if (threadIdx.x == 0 && blockIdx.x == 0)
         mergeProfileTotal[1] += clock64() - mergeProfile[1];
 }
 
 /* MERGEALLTYPES
  * Performs <runs> merge tests for each type at a given size
  */
 /*
 template<uint32_t blocks, uint32_t threads, uint32_t runs>
 void mergeAllTypes(uint64_t size) {
     PS("uint32_t", size)  mergeType<uint32_t, blocks, threads, runs>(size); printf("\n");
     PS("float", size)mergeType<float, blocks, threads, runs>(size);    printf("\n");
     PS("uint64_t", size)  mergeType<uint64_t, blocks, threads, runs>(size); printf("\n");
     PS("double", size)    mergeType<double, blocks, threads, runs>(size);   printf("\n");
 }
 */
 
 /* MAIN
  * Generates random arrays, merges them.
  */
 /*
 int main(int argc, char *argv[]) {
   #define blocks  112
   #define threads 128
   #define runs 10
     mergeAllTypes<blocks, threads, runs>(1000000);
     mergeAllTypes<blocks, threads, runs>(10000000);
     mergeAllTypes<blocks, threads, runs>(100000000);
 }
 */
 
 /* CUDAWORKLOADDIAGONALS
  * Performs a 32-wide binary search on one glboal diagonal per block to find the intersection with the path.
  * This divides the workload into independent merges for the next step
  */
 #define MAX(X,Y) (((X) > (Y)) ? (X) : (Y))
 #define MIN(X,Y) (((X) < (Y)) ? (X) : (Y))
 template <typename F>
 __device__ void cudaWorkloadDiagonals(F * copyFreq, int* copyIndex, int* copyIsLeaf,
                                       int cStart, int cEnd,
                                       F * iNodesFreq,
                                       int iStart, int iEnd, int iNodesCap,
                                       uint32_t * diagonal_path_intersections,
                                       /* Shared Memory */
                                       int32_t& x_top, int32_t& y_top, int32_t& x_bottom, int32_t& y_bottom,
                                       int32_t& found, int32_t* oneorzero) 
{
     uint32_t A_length = cEnd - cStart;
     uint32_t B_length = MOD(iEnd - iStart, iNodesCap);
     // Calculate combined index around the MergePath "matrix"
     int32_t combinedIndex = ((uint64_t)blockIdx.x * ((uint64_t)A_length + (uint64_t)B_length)) / (uint64_t)gridDim.x;
     /*
     __shared__ int32_t x_top, y_top, x_bottom, y_bottom,  found;
     __shared__ int32_t oneorzero[32];
     */
     int threadOffset = threadIdx.x - 16;
 
     if (threadIdx.x < 32) {
         // Figure out the coordinates of our diagonal
         if (A_length >= B_length) {
             x_top = MIN(combinedIndex, A_length);
             y_top = combinedIndex > A_length ? combinedIndex - (A_length) : 0;
             x_bottom = y_top;
             y_bottom = x_top;
         } else {
             y_bottom = MIN(combinedIndex, B_length);
             x_bottom = combinedIndex > B_length ? combinedIndex - (B_length) : 0;
             y_top = x_bottom;
             x_top = y_bottom;
         }
     }
 
     if (/*threadIdx.x == */0) {
         printf("Diagonal block %d: (%d, %d) to (%d, %d)\n", blockIdx.x, x_top, y_top, x_bottom, y_bottom);
     }
 
     found = 0;
 
     // Search the diagonal
     while(!found) {
         // Update our coordinates within the 32-wide section of the diagonal
         int32_t current_x = x_top - ((x_top - x_bottom) >> 1) - threadOffset;
         int32_t current_y = y_top + ((y_bottom - y_top) >> 1) + threadOffset;
         int32_t getfrom_x = current_x + cStart - 1;
         // Below statement is a more efficient, divmodless version of the following
         //int32_t getfrom_y = MOD(iStart + current_y, iNodesCap);
         int32_t getfrom_y = iStart + current_y;
             
         if (threadIdx.x < 32) {
             if (getfrom_y >= iNodesCap) getfrom_y -= iNodesCap;
 
             // Are we a '1' or '0' with respect to A[x] <= B[x]
             if(current_x > A_length || current_y < 0) {
                 oneorzero[threadIdx.x] = 0;
             } else if(current_y >= B_length || current_x < 1) {
                 oneorzero[threadIdx.x] = 1;
             } else {
                 oneorzero[threadIdx.x] = (copyFreq[getfrom_x] <= iNodesFreq[getfrom_y]) ? 1 : 0;
             }
         }
 
         __syncthreads();
 
         // If we find the meeting of the '1's and '0's, we found the
         // intersection of the path and diagonal
         if(threadIdx.x > 0 && threadIdx.x < 32 &&
            (oneorzero[threadIdx.x] != oneorzero[threadIdx.x-1])) {
             found = 1;
             diagonal_path_intersections[blockIdx.x] = current_x;
             diagonal_path_intersections[blockIdx.x + gridDim.x + 1] = current_y;
         }
 
         __syncthreads();
 
         // Adjust the search window on the diagonal
         if(threadIdx.x == 16) {
             if(oneorzero[31] != 0) {
                 x_bottom = current_x;
                 y_bottom = current_y;
             } else {
                 x_top = current_x;
                 y_top = current_y;
             }
         }
         __syncthreads();
     }
 
     // Set the boundary diagonals (through 0,0 and A_length,B_length)
     if(threadIdx.x == 0 && blockIdx.x == 0) {
         diagonal_path_intersections[0] = 0;
         diagonal_path_intersections[gridDim.x + 1] = 0;
         diagonal_path_intersections[gridDim.x] = A_length;
         diagonal_path_intersections[gridDim.x + gridDim.x + 1] = B_length;
     }
 }
 
 // Serial (ugh) merge
 template <typename F>
 __device__ void merge(F* copyFreq, int* copyIndex, int* copyIsLeaf, int cStart, int cEnd,
                       F* iNodesFreq, int iStart, int iEnd, int iNodesCap,
                       F* tempFreq, int* tempIndex, int* tempIsLeaf, int& tempLength) 
{
     int len = 0;
     int iterCopy = cStart, iterINodes = iStart;
 
     while (iterCopy < cEnd && MOD(iEnd - iterINodes, iNodesCap) > 0)
     {
         if (copyFreq[iterCopy] <= iNodesFreq[iterINodes])
         {
             tempFreq[len] = copyFreq[iterCopy];
             tempIndex[len] = copyIndex[iterCopy];
             tempIsLeaf[len] = copyIsLeaf[iterCopy];
             ++iterCopy;
         }
         else
         {
             tempFreq[len] = iNodesFreq[iterINodes];
             tempIndex[len] = iterINodes;
             tempIsLeaf[len] = 0;
             iterINodes = MOD(iterINodes + 1, iNodesCap);
         }
         ++len;
     }
 
     while (iterCopy < cEnd)
     {
         tempFreq[len] = copyFreq[iterCopy];
         tempIndex[len] = copyIndex[iterCopy];
         tempIsLeaf[len] = copyIsLeaf[iterCopy];
         ++iterCopy;
         ++len;
     }
     while (MOD(iEnd - iterINodes, iNodesCap) > 0)
     {
         tempFreq[len] = iNodesFreq[iterINodes];
         tempIndex[len] = iterINodes;
         tempIsLeaf[len] = 0;
         iterINodes = MOD(iterINodes + 1, iNodesCap);
         ++len;
     }
 
     tempLength = len;
 }
 
 
 /* CUDAMERGESINGLEPATH
  * Performs merge windows within a thread block from that block's global diagonal
  * intersection to the next
  */
 #define K 512
 #define PAD_SIZE 0
 template <typename F>
 __device__ void cudaMergeSinglePath(F * copyFreq, int* copyIndex, int* copyIsLeaf,
                                     int cStart, int cEnd,
                                     F * iNodesFreq,
                                     int iStart, int iEnd, int iNodesCap,
                                     uint32_t * diagonal_path_intersections,
                                     F* tempFreq, int* tempIndex, int* tempIsLeaf,
                                     int tempLength) 
{
     // Temporary Code -- Serial Merge Per Block
     if (threadIdx.x == 0) {
         // Boundaries
         int x_block_top = diagonal_path_intersections[blockIdx.x];
         int y_block_top = diagonal_path_intersections[blockIdx.x + gridDim.x + 1];
         int x_block_stop = diagonal_path_intersections[blockIdx.x + 1];
         int y_block_stop = diagonal_path_intersections[blockIdx.x + gridDim.x + 2];
 
         // Actual indexes
         int x_start = x_block_top + cStart;
         int x_end = x_block_stop + cStart;
         int y_start = MOD(iStart + y_block_top, iNodesCap);
         int y_end = MOD(iStart + y_block_stop, iNodesCap);
 
         int offset = x_block_top + y_block_top;
 
         int dummy; // Unused result
         merge(copyFreq, copyIndex, copyIsLeaf, x_start, x_end,
               iNodesFreq, y_start, y_end, iNodesCap,
               tempFreq + offset, tempIndex + offset, tempIsLeaf + offset, dummy);
         if (0) {
             printf("block: %d x: %d %d, y: %d %d, contrib: %d\n", blockIdx.x, x_block_top, x_block_stop, y_block_top, y_block_stop, dummy);
         }
     }
 
 //     int thread = blockIdx.x * blockDim.x + threadIdx.x;
 //     //uint32_t A_length = cEnd - cStart + 1;
 //     //uint32_t B_length = MOD(iEnd - iStart, iNodesCap);
 //     // Storage space for local merge window
 //     // A total of five arrays are allocated, three for copy and two for iNodes -- Plus padding
 //     __shared__ char all_shared[2 * ((K + 2) * sizeof(F)) + 3 * ((K + 2) * sizeof(int)) + 5 * PAD_SIZE];
 //     F* copyFreqShared = (F*)all_shared;
 //     F* iNodesFreqShared = (F*)(all_shared + ((K + 2) * sizeof(F)) + PAD_SIZE);
 //     int* copyIndexShared = (int*)(all_shared + 2 * ((K + 2) * sizeof(F)) + 2 * PAD_SIZE);
 //     int* copyIsLeafShared = (int*)(all_shared
 //                                    + 2 * ((K + 2) * sizeof(F)) + ((K + 2) * sizeof(int))
 //                                    + 3 * PAD_SIZE);
 //     int* iNodesIndexShared = (int*)(all_shared
 //                                     + 2 * ((K + 2) * sizeof(F)) + 2 * ((K + 2) * sizeof(int))
 //                                     + 4 * PAD_SIZE);
 
 //     __shared__ uint32_t x_block_top, y_block_top, x_block_stop, y_block_stop;
 
 //     // Pre-calculate reused indices
 //     uint32_t threadIdX4 = threadIdx.x + threadIdx.x;
 //     threadIdX4 = threadIdX4 + threadIdX4;
 //     uint32_t threadIdX4p1 = threadIdX4 + 1;
 //     uint32_t threadIdX4p2 = threadIdX4p1 + 1;
 //     uint32_t threadIdX4p3 = threadIdX4p2 + 1;
 //     uint32_t Ax, Bx;
 
 //     // Define global window and create sentinels
 //     switch(threadIdx.x) {
 //     case 0:
 //         x_block_top = diagonal_path_intersections[blockIdx.x];
 //         copyFreqShared[0] = getNegativeInfinity<F>();
 //         break;
 //     case 64:
 //         y_block_top = diagonal_path_intersections[blockIdx.x + gridDim.x + 1];
 //         copyFreqShared[K+1] = getPositiveInfinity<F>();
 //         break;
 //     case 32:
 //         x_block_stop = diagonal_path_intersections[blockIdx.x + 1];
 //         iNodesFreqShared[0] = getNegativeInfinity<F>();
 //         break;
 //     case 96:
 //         y_block_stop = diagonal_path_intersections[blockIdx.x + gridDim.x + 2];
 //         iNodesFreqShared[K+1] = getPositiveInfinity<F>();
 //         break;
 //     default:
 //         break;
 //     }
 
 //     /*
 //     --A;
 //     --B;
 //     */
 //     __syncthreads();
 
 //     // Construct and merge windows from diagonal_path_intersections[blockIdx.x]
 //     // to diagonal_path_intersections[blockIdx.x+1]
 //     while(((x_block_top < x_block_stop) || (y_block_top < y_block_stop))) {
 
 //         // Load current local window
 //         {
 //             //F * Atemp = A + x_block_top;
 //             //F * Btemp = B + y_block_top;
 //             int32_t sharedX = threadIdx.x+1;
 //             int32_t cIndexOffset = x_block_top - 1, iIndexOffset = y_block_top - 1;
 //             // Actual array access indices
 //             int32_t cIndex = cStart + cIndexOffset + sharedX;
 //             int32_t iIndex = iStart + iIndexOffset + sharedX;
 //             // Mod shortcut, since iIndex only increases, don't need expensive operation
 //             iIndex = (iIndex >= iNodesCap) ? iIndex - iNodesCap : iIndex;
 
 //             // 4 parts -- 512 elements
 // #pragma unroll
 //             for (int i = 0; i < 4; ++i) {
 //                 if (cIndex <= cEnd) {
 //                     copyFreqShared[sharedX] = copyFreq[cIndex];
 //                     copyIndexShared[sharedX] = copyIndex[cIndex];
 //                     copyIsLeafShared[sharedX] = copyIsLeaf[cIndex];
 //                 }
 //                 if (iIndex >= iStart || iIndex < iEnd) {
 //                     iNodesFreqShared[sharedX] = iNodesFreq[iIndex];
 //                     iNodesIndexShared[sharedX] = iIndex;
 //                 }
 //                 cIndex += blockDim.x;
 //                 iIndex += blockDim.x;
 //                 iIndex = (iIndex >= iNodesCap) ? iIndex - iNodesCap : iIndex;
 //                 sharedX += blockDim.x;
 //             }
 //         }
 
 //         // Make sure this is before the sync
 //         int32_t tIndexOffset = x_block_top + y_block_top;
 
 //         __syncthreads();
 
 //         // Binary search diagonal in the local window for path
 //         {
 //             int32_t offset = threadIdX4 >> 1;
 //             Ax = offset + 1;
 //             F * BSm1 = iNodesFreqShared + threadIdX4p2;
 //             F * BS = BSm1 + 1;
 //             while(true) {
 //                 offset = ((offset+1) >> 1);
 //                 if(copyFreqShared[Ax] > BSm1[~Ax]) {
 //                     if(copyFreqShared[Ax-1] <= BS[~Ax]) {
 //                         //Found it
 //                         break;
 //                     }
 //                     Ax -= offset;
 //                 } else {
 //                     Ax += offset;
 //                 }
 //             }
 //         }
 
 //         Bx = threadIdX4p2 - Ax;
 
 //         // Merge four elements starting at the found path intersection
 //         F copyFreqi = copyFreqShared[Ax];
 //         int copyIndexi = copyIndexShared[Ax];
 //         int copyIsLeafi = copyIsLeafShared[Ax];
 
 //         F iNodesFreqi = iNodesFreqShared[Bx];
 //         int iNodesIndexi = iNodesIndexShared[Bx];
 
 //         F tempFreqi;
 //         int tempIndexi;
 //         int tempIsLeafi;
 
 //         // Merge Step
 //         if (copyFreqi > iNodesFreqi) {
 //             tempFreqi = iNodesFreqi;
 //             tempIndexi = iNodesIndexi;
 //             tempIsLeafi = 0;
 //             ++Bx;
 //             iNodesFreqi = iNodesFreqShared[Bx];
 //             iNodesIndexi = iNodesIndexShared[Bx];
 //         } else {
 //             tempFreqi = copyFreqi;
 //             tempIndexi = copyIndexi;
 //             tempIsLeafi = copyIsLeafi;
 //             ++Ax;
 //             copyFreqi = copyFreqShared[Ax];
 //             copyIndexi = copyIndexShared[Ax];
 //             copyIsLeafi = copyIsLeafShared[Ax];
 //         }
 //         if (tIndexOffset + threadIdX4 < tempLength) {
 //             tempFreq[tIndexOffset + threadIdX4] = tempFreqi;
 //             tempIndex[tIndexOffset + threadIdX4] = tempIndexi;
 //             tempIsLeaf[tIndexOffset + threadIdX4] = tempIsLeafi;
 //         }
 
 //         // Merge Step
 //         if (copyFreqi > iNodesFreqi) {
 //             tempFreqi = iNodesFreqi;
 //             tempIndexi = iNodesIndexi;
 //             tempIsLeafi = 0;
 //             ++Bx;
 //             iNodesFreqi = iNodesFreqShared[Bx];
 //             iNodesIndexi = iNodesIndexShared[Bx];
 //         } else {
 //             tempFreqi = copyFreqi;
 //             tempIndexi = copyIndexi;
 //             tempIsLeafi = copyIsLeafi;
 //             ++Ax;
 //             copyFreqi = copyFreqShared[Ax];
 //             copyIndexi = copyIndexShared[Ax];
 //             copyIsLeafi = copyIsLeafShared[Ax];
 //         }
 //         if (tIndexOffset + threadIdX4p1 < tempLength) {
 //             tempFreq[tIndexOffset + threadIdX4p1] = tempFreqi;
 //             tempIndex[tIndexOffset + threadIdX4p1] = tempIndexi;
 //             tempIsLeaf[tIndexOffset + threadIdX4p1] = tempIsLeafi;
 //         }
 
 //         // Merge Step
 //         if (copyFreqi > iNodesFreqi) {
 //             tempFreqi = iNodesFreqi;
 //             tempIndexi = iNodesIndexi;
 //             tempIsLeafi = 0;
 //             ++Bx;
 //             iNodesFreqi = iNodesFreqShared[Bx];
 //             iNodesIndexi = iNodesIndexShared[Bx];
 //         } else {
 //             tempFreqi = copyFreqi;
 //             tempIndexi = copyIndexi;
 //             tempIsLeafi = copyIsLeafi;
 //             ++Ax;
 //             copyFreqi = copyFreqShared[Ax];
 //             copyIndexi = copyIndexShared[Ax];
 //             copyIsLeafi = copyIsLeafShared[Ax];
 //         }
 //         if (tIndexOffset + threadIdX4p2 < tempLength) {
 //             tempFreq[tIndexOffset + threadIdX4p2] = tempFreqi;
 //             tempIndex[tIndexOffset + threadIdX4p2] = tempIndexi;
 //             tempIsLeaf[tIndexOffset + threadIdX4p2] = tempIsLeafi;
 //         }
 
 //         // Merge Step -- Final
 //         if (copyFreqi > iNodesFreqi) {
 //             if (tIndexOffset + threadIdX4p3 < tempLength) {
 //                 tempFreq[tIndexOffset + threadIdX4p3] = iNodesFreqi;
 //                 tempIndex[tIndexOffset + threadIdX4p3] = iNodesIndexi;
 //                 tempIsLeaf[tIndexOffset + threadIdX4p3] = 0;
 //             }
 //         } else {
 //             if (tIndexOffset + threadIdX4p3 < tempLength) {
 //                 tempFreq[tIndexOffset + threadIdX4p3] = copyFreqi;
 //                 tempIndex[tIndexOffset + threadIdX4p3] = copyIndexi;
 //                 tempIsLeaf[tIndexOffset + threadIdX4p3] = copyIsLeafi;
 //             }
 //         }
 // /*
 //         F Ai, Bi, Ci;
 //         Ai = A_shared[Ax];
 //         Bi = B_shared[Bx];
 //         if(Ai > Bi) {Ci = Bi; Bx++; Bi = B_shared[Bx];} else {Ci = Ai; Ax++; Ai = A_shared[Ax];}
 //         Ctemp[threadIdX4] = Ci;
 //         if(Ai > Bi) {Ci = Bi; Bx++; Bi = B_shared[Bx];} else {Ci = Ai; Ax++; Ai = A_shared[Ax];}
 //         Ctemp[threadIdX4p1] = Ci;
 //         if(Ai > Bi) {Ci = Bi; Bx++; Bi = B_shared[Bx];} else {Ci = Ai; Ax++; Ai = A_shared[Ax];}
 //         Ctemp[threadIdX4p2] = Ci;
 //         Ctemp[threadIdX4p3] = Ai > Bi ? Bi : Ai;
 // */
 //         // Update for next window
 //         if(threadIdx.x == 127) {
 //             x_block_top += Ax - 1;
 //             y_block_top += Bx - 1;
 //         }
 
 //         __syncthreads();
 //     } // Go to next window
 
 }


// `unsigned int` versions
template
__device__ void parMerge<unsigned int>(unsigned int* copyFreq, int* copyIndex, int* copyIsLeaf, int cStart, int cEnd,
   unsigned int* iNodesFreq, int iStart, int iEnd, int iNodesCap,
   unsigned int* tempFreq, int* tempIndex, int* tempIsLeaf, int& tempLength,
   uint32_t* diagonal_path_intersections, int blocks, int threads,
   /* Shared Memory */
   int32_t& x_top, int32_t& y_top, int32_t& x_bottom, int32_t& y_bottom,
   int32_t& found, int32_t* oneorzero);

template
__device__ void merge<unsigned int>(unsigned int* copyFreq, int* copyIndex, int* copyIsLeaf, int cStart, int cEnd,
    unsigned int* iNodesFreq, int iStart, int iEnd, int iNodesCap,
    unsigned int* tempFreq, int* tempIndex, int* tempIsLeaf, int& tempLength);
 