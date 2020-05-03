#ifndef DEFLATE_CUH
#define DEFLATE_CUH

#include "__gpu_printing.cuh"
#include "huffman_host_device.hh"

template <typename Q, int PART_SIZE>
__global__ void print_deflated(Q* coded, size_t gid) {
    if (blockIdx.x * blockDim.x + threadIdx.x != gid) return;
    printf("print after deflating\n");
    //    for_each(coded, coded + PART_SIZE, [](Q& i) { print_by_type(i, '_', '\n'); });
    for (size_t i = 0; i < PART_SIZE; i++) {
        print_by_type(*(coded + i), '_', '\n');
    }
    printf("\n");
}

template <typename Q>
__global__ void deflate_v3(Q*      hcoded,  //
                           size_t  len,
                           size_t* densely_meta,
                           int     PART_SIZE) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (len - 1) / PART_SIZE + 1) return;
    uint8_t bitwidth;
    size_t  densely_coded_lsb_pos = sizeof(Q) * 8, total_bitwidth = 0;
    size_t  ending = (gid + 1) * PART_SIZE <= len ? PART_SIZE : len - gid * PART_SIZE;
//    if ((gid + 1) * PART_SIZE > len) printf("\n\ngid %lu\tending %lu\n\n", gid, ending);
    Q  msb_bw_coded_lsb, _1, _2;
    Q* current = hcoded + gid * PART_SIZE;
    for (size_t i = 0; i < ending; i++) {
        msb_bw_coded_lsb = hcoded[gid * PART_SIZE + i];
        bitwidth         = *((uint8_t*)&msb_bw_coded_lsb + (sizeof(Q) - 1));

        *((uint8_t*)&msb_bw_coded_lsb + sizeof(Q) - 1) = 0x0;
        if (densely_coded_lsb_pos == sizeof(Q) * 8) *current = 0x0;  // a new unit of data type
        if (bitwidth <= densely_coded_lsb_pos) {
            densely_coded_lsb_pos -= bitwidth;
            *current |= msb_bw_coded_lsb << densely_coded_lsb_pos;
#ifdef DEBUG
            if (gid == 0) {
                printf("%lu\tmsb pos: %lu\t", i, densely_coded_lsb_pos);
                printf("%p: \t", current);
                print_by_type(*current, '_', '\t');
                print_code_only(msb_bitwidth__coded_lsb, bitwidth, '\0');
            }
#endif
            if (densely_coded_lsb_pos == 0) {
                densely_coded_lsb_pos = sizeof(Q) * 8;
                ++current;
            }
        } else {
            // example: we have 5-bit code 11111 but 3 bits left for (*current)
            // we put first 3 bits of 11111 to the last 3 bits of (*current)
            // and put last 2 bits from MSB of (*(++current))
            // the comment continues with the example
            _1 = msb_bw_coded_lsb >> (bitwidth - densely_coded_lsb_pos);
            _2 = msb_bw_coded_lsb << (sizeof(Q) * 8 - (bitwidth - densely_coded_lsb_pos));
            *current |= _1;
#ifdef DEBUG
            if (gid == 0) {
                printf("%lu\tmsb pos: %lu\t", i, (size_t)0);
                printf("%p: \t", current);
                print_by_type(*current, '_', '\t');
                print_code_only(msb_bitwidth__coded_lsb, bitwidth, '\0');
            }
#endif
            *(++current) = 0x0;
            *current |= _2;
#ifdef DEBUG
            if (gid == 0) {
                printf("%lu\tmsb pos: %lu\t", i, sizeof(Q) * 8 - (bitwidth - densely_coded_lsb_pos));
                printf("%p: \t", current);
                print_by_type(*current, '_', '\t');
                print_code_only(msb_bitwidth__coded_lsb, bitwidth, '\0');
            }
#endif
            densely_coded_lsb_pos = sizeof(Q) * 8 - (bitwidth - densely_coded_lsb_pos);
        }
        total_bitwidth += bitwidth;
    }
    *(densely_meta + gid) = total_bitwidth;
}

// template <typename Q, int PART_SIZE>
//__global__ void deflate_v2(Q*      Huffman_coded,  //
//                           size_t  len,
//                           size_t* metadata_deflated) {
//    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//    if (gid >= (len - 1) / PART_SIZE + 1) return;
//    uint8_t bitwidth;
//    size_t  densely_coded_lsb_pos = sizeof(Q) * 8, total_bitwidth = 0;
//    size_t  ending = (gid + 1) * PART_SIZE <= len ? PART_SIZE : len - gid * PART_SIZE;
//    Q       msb_bitwidth__coded_lsb, first_half, second_half;
//    Q*      current = Huffman_coded + gid * PART_SIZE;
//    for (size_t i = 0; i < ending; i++) {
//        msb_bitwidth__coded_lsb                               = Huffman_coded[gid * PART_SIZE + i];
//        bitwidth                                              = *((uint8_t*)&msb_bitwidth__coded_lsb + (sizeof(Q) - 1));
//        *((uint8_t*)&msb_bitwidth__coded_lsb + sizeof(Q) - 1) = 0x0;
//        if (densely_coded_lsb_pos == sizeof(Q) * 8) *current = 0x0;  // a new unit of data type
//        if (bitwidth <= densely_coded_lsb_pos) {
//            densely_coded_lsb_pos -= bitwidth;
//            *current |= msb_bitwidth__coded_lsb << densely_coded_lsb_pos;
//#ifdef DEBUG
//            if (gid == 0) {
//                printf("%lu\tmsb pos: %lu\t", i, densely_coded_lsb_pos);
//                printf("%p: \t", current);
//                print_by_type(*current, '_', '\t');
//                print_code_only(msb_bitwidth__coded_lsb, bitwidth, '\0');
//            }
//#endif
//            if (densely_coded_lsb_pos == 0) {
//                densely_coded_lsb_pos = sizeof(Q) * 8;
//                ++current;
//            }
//        } else {
//            // example: we have 5-bit code 11111 but 3 bits left for (*current)
//            // we put first 3 bits of 11111 to the last 3 bits of (*current)
//            // and put last 2 bits from MSB of (*(++current))
//            // the comment continues with the example
//            first_half  = msb_bitwidth__coded_lsb >> (bitwidth - densely_coded_lsb_pos);
//            second_half = msb_bitwidth__coded_lsb << (sizeof(Q) * 8 - (bitwidth - densely_coded_lsb_pos));
//            *current |= first_half;
//#ifdef DEBUG
//            if (gid == 0) {
//                printf("%lu\tmsb pos: %lu\t", i, (size_t)0);
//                printf("%p: \t", current);
//                print_by_type(*current, '_', '\t');
//                print_code_only(msb_bitwidth__coded_lsb, bitwidth, '\0');
//            }
//#endif
//            *(++current) = 0x0;
//            *current |= second_half;
//#ifdef DEBUG
//            if (gid == 0) {
//                printf("%lu\tmsb pos: %lu\t", i, sizeof(Q) * 8 - (bitwidth - densely_coded_lsb_pos));
//                printf("%p: \t", current);
//                print_by_type(*current, '_', '\t');
//                print_code_only(msb_bitwidth__coded_lsb, bitwidth, '\0');
//            }
//#endif
//            densely_coded_lsb_pos = sizeof(Q) * 8 - (bitwidth - densely_coded_lsb_pos);
//        }
//        total_bitwidth += bitwidth;
//    }
//    *(metadata_deflated + gid) = total_bitwidth;
//}

#endif
