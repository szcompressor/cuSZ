/* 
 * Parallel Huffman Construction - Generates canonical forward codebook
 *
 * Based on S. Arash Ostadzadeh, B. Maryam Elahi, Zeinab Zeinalpour, M. Amir Moulavi, and Koen Bertels. “A Two-phase Practical Parallel
 *     Algorithm  for  Construction  of  Huffman  Codes”.  In:Proceedings  of  the  International  Conference  on  Parallel  and  Distributed
 *     Processing Techniques and Applications, PDPTA 2007, Las Vegas, Nevada, USA, June 25-28, 2007, Volume 1. Ed. by Hamid R.Arabnia. CSREA Press, 2007, pp. 284–291
 */

//
// By Cody Rivera, 5/2020
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "cuda_mem.cuh"
#include "cuda_error_handling.cuh"
#include "par_merge.cuh"
#include "par_huffman.cuh"
#include "dbg_gpu_printing.cuh"

// Mathematically correct mod
#define MOD(a,b) ((((a)%(b))+(b))%(b))

// Profiling
extern __device__ long mergeProfile[2];
extern __device__ long mergeProfileTotal[2];

//#define DEBUG_PARHUFF
// Parallel huffman code generation
template <typename F>
__global__ void parHuff::GPU_GenerateCL(F* histogram, F* CL, int size,
                                /* Global Arrays */
                                F* lNodesFreq, int* lNodesLeader,
                                F* iNodesFreq, int* iNodesLeader,
                                F* tempFreq, int* tempIsLeaf, int* tempIndex,
                                F* copyFreq, int* copyIsLeaf, int* copyIndex,
                                uint32_t* diagonal_path_intersections, int mblocks, int mthreads) {
    extern __shared__ int32_t shmem[];
    // Shared variables
    int32_t& x_top = shmem[0];
    int32_t& y_top = shmem[1];
    int32_t& x_bottom = shmem[2];
    int32_t& y_bottom = shmem[3];
    int32_t& found = shmem[4];
    int32_t* oneorzero = &shmem[5];

    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int i = thread; // Adaptation for easier porting
    auto current_grid = this_grid();

    // profiling
    if (thread == 0) {
        for (int i = 0; i < 10; ++i) {
            s[i] = 0;
        }
    }

    /* Initialization */
    if (thread < size) {
        lNodesLeader[i] = -1;
        CL[i] = 0;
    }
    
    if (thread == 0) {
        iNodesFront = 0;
        iNodesRear = 0;
        lNodesCur = 0;
        
        iNodesSize = 0;
    }
    current_grid.sync();

    /* While there is not exactly one internal node */
    while (lNodesCur < size || iNodesSize > 1)
    {
#ifdef DEBUG_PARHUFF
        //printf("Thread %d\n", i);
        if (thread == 0) {
            printf("iNodes:\n");
            for (int i = iNodesFront; i != iNodesRear; i = MOD(i + 1, size))
            {
                printf("%d %d\n", iNodesFreq[i], iNodesLeader[i]);
            }
        }
#endif
        if (thread == 0)
            st[0] = clock64();
        /* Combine two most frequent nodes on same level */
        if (thread == 0) {
            F midFreq[4];
            int midIsLeaf[4];
            for (int i = 0; i < 4; ++i) {
                midFreq[i] = UINT_MAX;
            }

            if (lNodesCur < size)
            {
                midFreq[0] = lNodesFreq[lNodesCur];
                midIsLeaf[0] = 1;
            }
            if (lNodesCur < size - 1)
            {
                midFreq[1] = lNodesFreq[lNodesCur + 1];
                midIsLeaf[1]= 1;
            }
            if (iNodesSize >= 1)
            {
                midFreq[2] = iNodesFreq[iNodesFront];
                midIsLeaf[2] = 0;
            }
            if (iNodesSize >= 2)
            {
                midFreq[3] = iNodesFreq[MOD(iNodesFront + 1, size)];
                midIsLeaf[3] = 0;
            }


            /* Select the minimum of minimums - 4elt sorting network */
            /* TODO There's likely a good 1-warp faster way to do this */
            {
                F tempFreq;
                int tempIsLeaf;
                if (midFreq[1] > midFreq[3])
                {
                    tempFreq = midFreq[1];
                    midFreq[1] = midFreq[3];
                    midFreq[3] = tempFreq;
                    tempIsLeaf = midIsLeaf[1];
                    midIsLeaf[1] = midIsLeaf[3];
                    midIsLeaf[3] = tempIsLeaf;
                }
                if (midFreq[0] > midFreq[2])
                {
                    tempFreq = midFreq[0];
                    midFreq[0] = midFreq[2];
                    midFreq[2] = tempFreq;
                    tempIsLeaf = midIsLeaf[0];
                    midIsLeaf[0] = midIsLeaf[2];
                    midIsLeaf[2] = tempIsLeaf;
                }
                if (midFreq[0] > midFreq[1])
                {
                    tempFreq = midFreq[0];
                    midFreq[0] = midFreq[1];
                    midFreq[1] = tempFreq;
                    tempIsLeaf = midIsLeaf[0];
                    midIsLeaf[0] = midIsLeaf[1];
                    midIsLeaf[1] = tempIsLeaf;
                }
                if (midFreq[2] > midFreq[3])
                {
                    tempFreq = midFreq[2];
                    midFreq[2] = midFreq[3];
                    midFreq[3] = tempFreq;
                    tempIsLeaf = midIsLeaf[2];
                    midIsLeaf[2] = midIsLeaf[3];
                    midIsLeaf[3] = tempIsLeaf;
                }
                if (midFreq[1] > midFreq[2])
                {
                    tempFreq = midFreq[1];
                    midFreq[1] = midFreq[2];
                    midFreq[2] = tempFreq;
                    tempIsLeaf = midIsLeaf[1];
                    midIsLeaf[1] = midIsLeaf[2];
                    midIsLeaf[2] = tempIsLeaf;
                }
            }


#ifdef DEBUG_PARHUFF
            printf("mid:\n");
            for (int i = 0; i != 4; ++i)
            {
                printf("%d %d\n", midFreq[i], midIsLeaf[i]);
            }
#endif

            minFreq = midFreq[0];
            if (midFreq[1] < UINT_MAX)
            {
                minFreq += midFreq[1];
            }
            iNodesFreq[iNodesRear] = minFreq;
            iNodesLeader[iNodesRear] = -1;

            /* If is leaf */
            if (midIsLeaf[0])
            {
                lNodesLeader[lNodesCur] = iNodesRear;
                ++CL[lNodesCur]; ++lNodesCur;
            }
            else
            {
                iNodesLeader[iNodesFront] = iNodesRear;
                iNodesFront = MOD(iNodesFront + 1, size);
            }
            if (midIsLeaf[1])
            {
                lNodesLeader[lNodesCur] = iNodesRear;
                ++CL[lNodesCur]; ++lNodesCur;
            }
            else
            {
                iNodesLeader[iNodesFront] = iNodesRear;
                iNodesFront = MOD(iNodesFront + 1, size); /* ? */
            }

            //iNodesRear = MOD(iNodesRear + 1, size);

            iNodesSize = MOD(iNodesRear - iNodesFront, size);
        }

        if (thread == 0)
            s[0] = s[0] + (clock64() - st[0]);

        //int curLeavesNum;
        /* Select elements to copy -- parallelized */
        if (thread == 0)
            st[1] = clock64();

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
                copyFreq[i - lNodesCur] = lNodesFreq[i];
                copyIndex[i - lNodesCur] = i;
                copyIsLeaf[i - lNodesCur] = 1;
            }
        }

        current_grid.sync();
        if (thread == 0)
            s[1] = s[1] + (clock64() - st[1]);

        /* Updates Iterators */
        if (thread == 0)
            st[2] = clock64();
        if (thread == 0) {
            mergeRear = iNodesRear;
            mergeFront = iNodesFront;

            if ((curLeavesNum + iNodesSize) % 2 == 0)
            {
                iNodesFront = iNodesRear;
            }
            /* Odd number of nodes to merge - leave out one*/
            else if ((iNodesSize != 0)
                        && (curLeavesNum == 0 || (histogram[lNodesCur + curLeavesNum]
                                                <= iNodesFreq[MOD(iNodesRear - 1, size)])))
            {
                mergeRear = MOD(mergeRear - 1, size);
                iNodesFront = MOD(iNodesRear - 1, size);
            }
            else
            {
                iNodesFront = iNodesRear;
                --curLeavesNum;
            }

            lNodesCur = lNodesCur + curLeavesNum;
            iNodesRear = MOD(iNodesRear + 1, size);

#ifdef DEBUG_PARHUFF
            printf("adjusted iNodes:\n");
            for (int i = iNodesFront; i != iNodesRear; i = MOD(i + 1, size))
            {
                printf("%d %d\n", iNodesFreq[i], iNodesLeader[i]);
            }
            printf("minfreq %d\n", minFreq);
            printf("curleavesnum %d\n", curLeavesNum);
            printf("lnodescur %d\n", lNodesCur);

            printf("copy:\n");
            for (int i = 0; i != curLeavesNum; ++i)
            {
                printf("%d %d %d\n", copyFreq[i], copyIndex[i], copyIsLeaf[i]);
            }
#endif
        }
        current_grid.sync();
        if (thread == 0)
            s[2] = s[2] + (clock64() - st[2]);

        /* Parallelized */
        if (thread == 0)
            st[3] = clock64();
        /* Merging phase */
        
        
        /*if (thread == 0) {
            merge(copyFreq, copyIndex, copyIsLeaf, 0, curLeavesNum,
                    iNodesFreq, mergeFront, mergeRear, size,
                    tempFreq, tempIndex, tempIsLeaf, tempLength);
                    }*/

        parMerge(copyFreq, copyIndex, copyIsLeaf, 0, curLeavesNum,
                    iNodesFreq, mergeFront, mergeRear, size,
                    tempFreq, tempIndex, tempIsLeaf, tempLength,
                    diagonal_path_intersections, mblocks, mthreads,
                    x_top, y_top, x_bottom, y_bottom, found, oneorzero);
        
        if (thread == 0) {
#ifdef DEBUG_PARHUFF
            printf("temp:\n");
            for (int i = 0; i != tempLength; ++i)
            {
                printf("%d %d %d\n", tempFreq[i], tempIndex[i], tempIsLeaf[i]);
            }
#endif
        }
        current_grid.sync();
        if (thread == 0)
            s[3] = s[3] + (clock64() - st[3]);

        /* Melding phase -- New */
        if (thread < tempLength / 2) {
            int ind = MOD(iNodesRear + i, size);
            iNodesFreq[ind] = tempFreq[(2 * i)] + tempFreq[(2 * i) + 1];
            iNodesLeader[ind] = -1;

            if (tempIsLeaf[(2 * i)])
            {
                lNodesLeader[tempIndex[(2 * i)]] = ind;
                ++CL[tempIndex[(2 * i)]];
            }
            else
            {
                iNodesLeader[tempIndex[(2 * i)]] = ind;
            }
            if (tempIsLeaf[(2 * i) + 1])
            {
                lNodesLeader[tempIndex[(2 * i) + 1]] = ind;
                ++CL[tempIndex[(2 * i) + 1]];
            }
            else
            {
                iNodesLeader[tempIndex[(2 * i) + 1]] = ind;
            }
        }
        current_grid.sync();

        if (thread == 0) {
            iNodesRear = MOD(iNodesRear + (tempLength / 2), size);
        }
        current_grid.sync();


        /* Update leaders */
        if (thread < size) {
            if (lNodesLeader[i] != -1)
            {
                if (iNodesLeader[lNodesLeader[i]] != -1)
                {
                    lNodesLeader[i] = iNodesLeader[lNodesLeader[i]];
                    ++CL[i];
                }
            }
        }
        current_grid.sync();

        if (thread == 0) {
            iNodesSize = MOD(iNodesRear - iNodesFront, size);
        }
        current_grid.sync();
    }
    if (thread == 0) {
        for (int i = 0; i < 4; ++i) {
            printf("r%d: %lld\n", i, s[i]);
        }
        for (int i = 0; i < 2; ++i) {
            printf("mr%d: %ld\n", i, mergeProfileTotal[i]);
        }
    }
}


// "Locals" for GenerateCW
__device__ int CCL;
__device__ int CDPI;
__device__ int newCDPI;

// Parallelized with atomic writes, but could replace with Jiannan's similar code
template<typename F, typename H>
__global__ void parHuff::GPU_GenerateCW(F* CL, H* CW, int size)
{
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int i = thread; // Porting convenience
    auto current_grid = this_grid();

    /* Reverse in place - Probably a more CUDA-appropriate way */
    if (thread < size / 2) {
        F temp = CL[i];
        CL[i] = CL[size - i - 1];
        CL[size - i - 1] = temp;
    }
    current_grid.sync();

    if (thread == 0) {
        CCL = CL[0];
        CW[0] = 0;
        CDPI = 0;
        newCDPI = size - 1;
    }
    current_grid.sync();

    while (CDPI < size - 1)
    {

        if (i < size) {
            // Parallel canonical codeword generation
            if (i > CDPI && CL[i] == CCL) {
                CW[i] = CW[CDPI] + (i - CDPI);
            }

            // CDPI update
            if (i < size - 1 && CL[i + 1] > CCL) {
                atomicMin(&newCDPI, i);
            }
        }
        current_grid.sync();
        CDPI = newCDPI;

        if (thread == 0) {
            if (CDPI < size - 1)
            {
                int CLDiff = CL[CDPI + 1] - CL[CDPI];
                CW[CDPI + 1] = ((CW[CDPI] + 1) << CLDiff);
                CCL = CL[CDPI + 1];

                ++CDPI;
            }
        }
        newCDPI = size - 1;
        current_grid.sync();
    }
    current_grid.sync();

    if (thread < size) {
        /* Make encoded codeword compatible with CUSZ */
        //printf("%d: %d %x %x %x\n", i, CL[i], (((H)CL[i] & (H)0xffu) << ((sizeof(H) * 8) - 8)), CW[i], CW[i] | (((H)CL[i] & (H)0xffu) << ((sizeof(H) * 8) - 8)));
        CW[i] = CW[i] | (((H)CL[i] & (H)0xffu) << ((sizeof(H) * 8) - 8));
    }
    current_grid.sync();

    /* Reverse in place - This is only needed for debug purposes */
    if (thread < size / 2) {
        F temp = CL[i];
        CL[i] = CL[size - i - 1];
        CL[size - i - 1] = temp;
    }

    /* Reverse in place */
    if (thread < size / 2) {
        H temp = CW[i];
        CW[i] = CW[size - i - 1];
        CW[size - i - 1] = temp;
    }
}

// Helper implementations
template <typename T>
__global__ void GPU_FillArraySequence(T* array, unsigned int size) {
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread < size) {
        array[thread] = thread;
    }
}

// Precondition -- Result is preset to be equal to size
template <typename T>
__global__ void GPU_GetFirstNonzeroIndex(T* array, unsigned int size, unsigned int* result) {
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (array[thread] != 0) {
        atomicMin(result, thread);
    }
}

// Reorders given a set of indices. Programmer must ensure that all index[i]
// are unique or else race conditions may occur
template <typename T, typename Q>
__global__ void GPU_ReorderByIndex(T* array, Q* index, unsigned int size) {
    unsigned int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
    T temp;
    Q newIndex;
    if (thread < size) {
        temp     = array[thread];
        newIndex = index[thread];
        array[newIndex] = temp;
    }
}

// Parallel codebook generation wrapper
template <typename H>
void ParGetCodebook(int dict_size, unsigned int* _d_freq, H* _d_codebook) {
    auto _d_qcode = mem::CreateCUDASpace<int>(dict_size);

    // Sort Qcodes by frequency
    int nblocks = (dict_size / 1024) + 1;
    GPU_FillArraySequence<int><<<nblocks, 1024>>>(_d_qcode, (unsigned int) dict_size);
    cudaDeviceSynchronize();

    SortByFreq(_d_freq, _d_qcode, dict_size);
    cudaDeviceSynchronize();

    unsigned int* d_first_nonzero_index;
    unsigned int first_nonzero_index = dict_size;
    cudaMalloc(&d_first_nonzero_index, sizeof(unsigned int));
    cudaMemcpy(d_first_nonzero_index, &first_nonzero_index, sizeof(unsigned int), cudaMemcpyHostToDevice);
    GPU_GetFirstNonzeroIndex<unsigned int><<<nblocks, 1024>>>(_d_freq, dict_size, d_first_nonzero_index);
    cudaDeviceSynchronize();
    cudaMemcpy(&first_nonzero_index, d_first_nonzero_index, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_first_nonzero_index);

    int nz_dict_size = dict_size - first_nonzero_index;
    unsigned int* _nz_d_freq = _d_freq + first_nonzero_index;
    H* _nz_d_codebook = _d_codebook + first_nonzero_index;
    int nz_nblocks = (nz_dict_size / 1024) + 1;

    // Memory Allocation -- Perhaps put in another wrapper
    unsigned int* CL = nullptr;
    /*unsigned int* lNodesFreq*/           int* lNodesLeader = nullptr;                                              
    unsigned int* iNodesFreq = nullptr;    int* iNodesLeader = nullptr;                                              
    unsigned int* tempFreq = nullptr;      int* tempIsLeaf = nullptr;    int* tempIndex = nullptr;
    unsigned int* copyFreq = nullptr;      int* copyIsLeaf = nullptr;    int* copyIndex = nullptr;
    cudaMalloc(&CL, nz_dict_size * sizeof(unsigned int));
    cudaMalloc(&lNodesLeader, nz_dict_size * sizeof(int));
    cudaMalloc(&iNodesFreq, nz_dict_size * sizeof(unsigned int));
    cudaMalloc(&iNodesLeader, nz_dict_size * sizeof(int));
    cudaMalloc(&tempFreq, nz_dict_size * sizeof(unsigned int));
    cudaMalloc(&tempIsLeaf, nz_dict_size * sizeof(int));
    cudaMalloc(&tempIndex, nz_dict_size * sizeof(int));
    cudaMalloc(&copyFreq, nz_dict_size * sizeof(unsigned int));
    cudaMalloc(&copyIsLeaf, nz_dict_size * sizeof(int));
    cudaMalloc(&copyIndex, nz_dict_size * sizeof(int));
    cudaMemset(CL, 0, nz_dict_size * sizeof(int));

    // Merge configuration -- Change for V100
    int ELTS_PER_SEQ_MERGE = 16;
    
    int mblocks = (nz_dict_size / ELTS_PER_SEQ_MERGE) + 1;
    int mthreads = 32;
    uint32_t* diagonal_path_intersections;
    cudaMalloc(&diagonal_path_intersections, (2 * (mblocks + 1)) * sizeof(uint32_t));
    
    // Codebook already initted
    cudaDeviceSynchronize();

    // Call first kernel
    // Collect arguments
    void* CL_Args[] = {
        (void *)&_nz_d_freq,
        (void *)&CL,
        (void *)&nz_dict_size,
        (void *)&_nz_d_freq,
        (void *)&lNodesLeader,
        (void *)&iNodesFreq,
        (void *)&iNodesLeader,
        (void *)&tempFreq,
        (void *)&tempIsLeaf,
        (void *)&tempIndex,
        (void *)&copyFreq,
        (void *)&copyIsLeaf,
        (void *)&copyIndex,
        (void *)&diagonal_path_intersections,
        (void *)&mblocks,
        (void *)&mthreads
    };
    // Cooperative Launch
    cudaLaunchCooperativeKernel((void *)parHuff::GPU_GenerateCL<unsigned int>,
                                mblocks,
                                mthreads,
                                CL_Args,
                                5 * sizeof(int32_t) + 32 * sizeof(int32_t));
    /*
    parHuff:GPU_GenerateCL<unsigned int><<<nz_nblocks, 1024>>>(_nz_d_freq, CL, nz_dict_size,
                                                                    _nz_d_freq, lNodesLeader, 
                                                                    iNodesFreq, iNodesLeader,
                                                                    tempFreq, tempIsLeaf, tempIndex,
                                                                    copyFreq, copyIsLeaf, copyIndex,                                                                
                                                                    diagonal_path_intersections, mblocks, mthreads);
    */
    cudaDeviceSynchronize();

    void* CW_Args[] = {
        (void *)&CL,
        (void *)&_nz_d_codebook,
        (void *)&nz_dict_size
    };

    // Call second kernel
    cudaLaunchCooperativeKernel((void *)parHuff::GPU_GenerateCW<unsigned int, H>,
                                nz_nblocks,
                                1024,
                                CW_Args);
    /*
    parHuff::GPU_GenerateCW<unsigned int, H><<<nz_nblocks, 1024>>>(CL, _nz_d_codebook, nz_dict_size);
    */
    cudaDeviceSynchronize();

    #ifdef D_DEBUG_PRINT
    print_codebook<H><<<1, 32>>>(_d_codebook, dict_size);  // PASS
    cudaDeviceSynchronize();
    #endif

    GPU_ReorderByIndex<H, int><<<nblocks, 1024>>>(_d_codebook, _d_qcode, (unsigned int) dict_size);
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
    cudaFree(_d_qcode);
    cudaFree(diagonal_path_intersections);
    cudaDeviceSynchronize();

    #ifdef D_DEBUG_PRINT
    print_codebook<H><<<1, 32>>>(_d_codebook, dict_size);  // PASS
    cudaDeviceSynchronize();
    #endif
}

// Specialize wrapper
template void ParGetCodebook<uint32_t>(int dict_size, unsigned int* freq, uint32_t* codebook);
template void ParGetCodebook<uint64_t>(int dict_size, unsigned int* freq, uint64_t* codebook);
