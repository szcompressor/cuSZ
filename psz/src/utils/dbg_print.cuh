#ifndef UTILS_DBG_PRINT_CUH
#define UTILS_DBG_PRINT_CUH

/**
 * @file dbg_print.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2020-09-20
 * Created on 2020-03-17
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

template <typename Q, int PART_SIZE>
__global__ void print_deflated(Q* coded, size_t gid)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != gid) return;
    printf("print after deflating\n");
    //    for_each(coded, coded + PART_SIZE, [](Q& i) { print_by_type(i, '_', '\n'); });
    for (size_t i = 0; i < PART_SIZE; i++) { print_by_type(*(coded + i), '_', '\n'); }
    printf("\n");
}

template <typename T>
__global__ void print_histogram(T* freq, size_t size, size_t radius = 20)
{
    const int DICT_SIZE = size; /* Dynamic sizing */
    if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
        for (size_t i = DICT_SIZE / 2 - radius; i < DICT_SIZE / 2 + radius; i++) {
            if (i % 10 == 0) printf("\n");
            printf("%4lu: %-12lu", i, static_cast<size_t>(freq[i]));
        }
        printf("\n");
    }
}

template <typename T>
__device__ __host__ void print_by_type(T num, char sep = '_', char ending = '\n')
{
    for (size_t j = 0; j < sizeof(T) * CHAR_BIT; j++) {
        printf("%u", (num >> ((sizeof(T) * CHAR_BIT - 1) - j)) & 0x01u);
        if (j != 0 and j != sizeof(T) * CHAR_BIT - 1 and j % 8 == 7) printf("%c", sep);
    }
    printf("%c", ending);
}

// MSB to LSB
template <typename T>
__device__ __host__ void print_code_only(T num, size_t bitwidth, char sep = '_', char ending = '\n')
{
    for (size_t j = 0; j < bitwidth; j++) {
        printf("%u", (num >> ((bitwidth - 1) - j)) & 0x01u);
        if (j != 0 and j != bitwidth - 1 and j % 8 == 7) printf("%c", sep);
    }
    printf("%c", ending);
}

template <typename T>
__device__ __host__ void snippet_print_bitset_full(T num)
{
    print_by_type(num, '_', '\t');
    size_t bitwidth = *((uint8_t*)&num + sizeof(T) - 1);
    //    size_t code_bitwidth = ((static_cast<T>(0xffu) << (sizeof(T) * 8 - 8)) & num) >> (sizeof(T) * 8 - 8);
    printf("len: %3lu\tcode: ", bitwidth);
    print_code_only<T>(num, bitwidth, '\0', '\n');
}

template <typename T>
__global__ void print_codebook(T* codebook, size_t len)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    printf("--------------------------------------------------------------------------------\n");
    printf("printing codebook\n");
    printf("--------------------------------------------------------------------------------\n");
    __shared__ T buffer;
    for (size_t i = 0; i < len; i++) {
        buffer = codebook[i];
        if (buffer == ~((T)0x0)) continue;
        printf("%5lu\t", i);
        snippet_print_bitset_full(buffer);
    }
    printf("--------------------------------------------------------------------------------\n");
    printf("done printing codebook\n");
    printf("--------------------------------------------------------------------------------\n");
}

template <typename T>
__global__ void get_entropy(T* freq)
{
}

// TODO real GPU version
template <typename T, typename Q>
__global__ void get_theoretical_dense_Huffman_coded_length(T* codebook, Q* freq, size_t codebook_len)
{
}

// template <typename T>
//__global__ void print_Huffman_coded_before_deflating(T* coded, size_t len=200) {
//    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
//    printf("print Huffman coded before it is deflated\n");
//    for (size_t i = 0; i < 200; i++) {
//        if (coded[i] == ~((T)0x0)) continue;
//        printf("%5lu\t", i);
//        snippet_print_bitset_full(coded[i]);
//    }
//    printf("\n");
//}

template <typename T>
__global__ void print_Huffman_coded_before_deflating(T* coded, size_t len)
{
    if (blockIdx.x != 0) return;
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (coded[gid] == ~((T)0x0)) return;
    printf("%5lu\t", gid);
    snippet_print_bitset_full(coded[gid]);

    //        if (coded[i] == ~((T)0x0)) continue;
    //    printf("print Huffman coded before it is deflated\n");
    //    for (size_t i = 0; i < 200; i++) {
    //        if (coded[i] == ~((T)0x0)) continue;
    //        printf("%5lu\t", i);
    //        snippet_print_bitset_full(coded[i]);
    //    }
    //    printf("\n");
}

#endif