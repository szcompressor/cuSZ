//
// Created by jtian on 3/19/20.
//

#ifndef DEFLATING_CPU_HH
#define DEFLATING_CPU_HH

#include <cstdlib>

//template <typename Q, int PART_SIZE>
//void deflate_v2_cpu(Q*      Huffman_coded,  //
//                    size_t  len,
//                    size_t* metadata_deflated,
//                    size_t  gid) {
//    // TODO assert does not work well
//    //    assert(sizeof(Q) == 8);
//    if (gid >= (len - 1) / PART_SIZE + 1) return;
//
//    uint8_t bitwidth;
//    size_t  Huffman_code_msb_pos = 64, total_bitwidth = 0;
//    Q       coded_and_bitwidth, coded, coded_second;
//    Q*      current = Huffman_coded + gid * PART_SIZE;
//    size_t ending = (gid + 1) * PART_SIZE <= len ? PART_SIZE : len - gid * PART_SIZE;
//
//    for (size_t i = 0; i < ending; i++) {
//        coded_and_bitwidth = Huffman_coded[gid * PART_SIZE + i];
//        bitwidth           = ((static_cast<Q>(0xffu) << (sizeof(Q) * 8 - 8)) & coded_and_bitwidth) >> (sizeof(Q) * 8 - 8);
//        Q mask_code        = (~(static_cast<Q>(0x0u)) << 8u) >> 8u;
//        coded              = (coded_and_bitwidth & mask_code);  // clear up bits indicating bitwidth
//
//        if (bitwidth <= Huffman_code_msb_pos) {
//            Huffman_code_msb_pos -= bitwidth;
//            *current |= coded << (Huffman_code_msb_pos);
//            //            if (gid == 0) {
//            //                printf("%p: ", current);
//            //                print_by_type(*current, '_', '\n');
//            //            }
//            if (Huffman_code_msb_pos == 0) {
//                Huffman_code_msb_pos = 64;
//                ++current;
//                //                if (gid == 0) {
//                //                    printf("\nbefore cleanup current unit\n%p: ", current);
//                //                    print_by_type(*current, '_', '\n');
//                //                }
//                *current = 0x0;
//                //                if (gid == 0) {
//                //                    printf("cleanup current unit\n%p: ", current);
//                //                    print_by_type(*current, '_', '\n');
//                //                    printf("\n");
//                //                }
//            }
//        } else {
//            // example: we have 5-bit code 11111 but 3 bits left for (*current)
//            // we put first 3 bits of 11111 to the last 3 bits of (*current)
//            // and put last 2 bits from MSB of (*(++current))
//            // the comment continues with the example
//            // TODO should not be explicit uint64_t due to templating
//            uint64_t mask_cross_bytes = ~(0xffffffffffffffff << (bitwidth - Huffman_code_msb_pos));  // the mask_cross_bytes is 0b00...00011
//            coded_second = (coded & mask_cross_bytes) << (64 - (bitwidth - Huffman_code_msb_pos));   // we get the last (5-3) bits out of 11111
//            coded >>= (bitwidth - Huffman_code_msb_pos);                                             // we need the first 3 bits out of 11111 near LSB
//            *current |= coded;
//            //            if (gid == 0) {
//            //                printf("%p: ", current);
//            //                print_by_type(*current, '_', '\n');
//            //            }
//            ++current;
//            //            if (gid == 0) {
//            //                printf("\nbefore cleanup current unit\n%p: ", current);
//            //                print_by_type(*current, '_', '\n');
//            //            }
//            *current = 0x0;
//            //            if (gid == 0) {
//            //                printf("cleanup current unit\n%p: ", current);
//            //                print_by_type(*current, '_', '\n');
//            //                printf("\n");
//            //            }
//            *current |= coded_second;
//            Huffman_code_msb_pos = 64 - (bitwidth - Huffman_code_msb_pos);
//            //            if (gid == 0) {
//            //                printf("%p: ", current);
//            //                print_by_type(*current, '_', '\n');
//            //            }
//        }
//        total_bitwidth += bitwidth;
//    }
//
//    // TODO change to uint8_t (reinterpret)
//    if (Huffman_code_msb_pos != 0) {
//        *current &= 0xffffffffffffffff << Huffman_code_msb_pos;
//    }
//    *(metadata_deflated + gid) = total_bitwidth;
//}

#endif  // HUFFMAN_PROJ_DEFLATING_CPU_HH
