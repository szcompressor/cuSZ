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

#ifndef PAR_MERGE_CUH
#define PAR_MERGE_CUH

#include <cuda.h>
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

using namespace cooperative_groups;

// Partition array
template <typename F>
__device__ void cudaWorkloadDiagonals(F * copyFreq, int* copyIndex, int* copyIsLeaf,
    int cStart, int cEnd,
    F * iNodesFreq,
    int iStart, int iEnd, int iNodesCap,
    uint32_t * diagonal_path_intersections,
    /* Shared Memory */
    int32_t& x_top, int32_t& y_top, int32_t& x_bottom, int32_t& y_bottom,
    int32_t& found, int32_t* oneorzero);
    
// Merge partitions
template <typename F>
__device__ void cudaMergeSinglePath(F * copyFreq, int* copyIndex, int* copyIsLeaf,
    int cStart, int cEnd,
    F * iNodesFreq,
    int iStart, int iEnd, int iNodesCap,
    uint32_t * diagonal_path_intersections,
    F* tempFreq, int* tempIndex, int* tempIsLeaf,
    int tempLength);

template <typename F>
__device__ void parMerge(F* copyFreq, int* copyIndex, int* copyIsLeaf, int cStart, int cEnd,
    F* iNodesFreq, int iStart, int iEnd, int iNodesCap,
    F* tempFreq, int* tempIndex, int* tempIsLeaf, int& tempLength,
    uint32_t* diagonal_path_intersections, int blocks, int threads,
    /* Shared Memory */
    int32_t& x_top, int32_t& y_top, int32_t& x_bottom, int32_t& y_bottom,
    int32_t& found, int32_t* oneorzero);

template <typename F>
__device__ void merge(F* copyFreq, int* copyIndex, int* copyIsLeaf, int cStart, int cEnd,
    F* iNodesFreq, int iStart, int iEnd, int iNodesCap,
    F* tempFreq, int* tempIndex, int* tempIsLeaf, int& tempLength);
    
#endif
