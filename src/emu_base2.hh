//
// Created by JianNan Tian on 2019-08-28.
//

#ifndef BASE2_SPECIFIC_HH
#define BASE2_SPECIFIC_HH

#include <cstdint>

#include "deprecated_constants.hh"

typedef union fp32_pun {
    float    fp32_num;
    uint32_t fp32_bits;
} fp32_pun_t;

typedef union fp64_pun {
    double   fp64_num;
    uint64_t fp64_bits;
} fp64_pun_t;

namespace bitwise {  // TODO: FP32

inline void divided_by_in_exp(fp64_pun_t* pun, int base2_exp) {
    /* precision = 2^{-base2_exp}, e.g., 2^{-10}
     * 1) reset exponent, &= 0x800f'ffff'ffff'ffffLLU (FP64_RES_EXPO_MASK)
     *    0b 1 00000000 1111111111...1111111111 LLU
     *       s exponent --- 23-bit mantissa ---
     * 2) set exponent, |= ???????? << FP64_MANTISSA_WIDTH
     *    0b 0 ???????? 0000000000...0000000000 LLU
     */
    u_int64_t exponent = (pun->fp64_bits >> FP64_MANTISSA_WIDTH) & FP64_EXPONENT_MASK;
    exponent -= base2_exp; /* division equivalent */
    pun->fp64_bits &= FP64_RES_EXPO_MASK;
    pun->fp64_bits |= exponent << FP64_MANTISSA_WIDTH;
}

inline void multiplied_by_in_exp(fp64_pun_t* pun, int exp_base2) {
    u_int64_t exponent = (pun->fp64_bits >> FP64_MANTISSA_WIDTH) & FP64_EXPONENT_MASK;
    exponent += exp_base2;  // division
    pun->fp64_bits &= FP64_RES_EXPO_MASK;
    pun->fp64_bits |= exponent << FP64_MANTISSA_WIDTH;
}

inline u_int64_t get_exponent(fp64_pun_t* pun) {
    return (pun->fp64_bits >> FP64_MANTISSA_WIDTH) & FP64_EXPONENT_MASK;
}

inline u_int64_t get_signum(fp64_pun_t* pun) {
    return pun->fp64_bits >> FP64_ABSOLUTE_WIDTH;
}

inline void update_signum(fp64_pun* pun, u_int64_t _sign) {
    /*
     * 1) reset signum, &= 0x7fff'ffff'ffff'ffffLLU (FP64_RES_EXPO_MASK)
     *    0b 0 11111111111 1111111111...1111111111 LLU
     *       s exponent--> <-- 52-bit mantissa -->
     * 2) set signum, |= ???????? << FP64_MANTISSA_WIDTH
     *    0b ? 00000000000 0000000000...0000000000 LLU
     */
    pun->fp64_bits &= FP64_RES_SIGN_MASK;
    pun->fp64_bits |= _sign << FP64_ABSOLUTE_WIDTH;
}

}  // namespace bitwise

#endif  // BASE2_SPECIFIC_HH
