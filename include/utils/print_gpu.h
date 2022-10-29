/**
 * @file print.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-28
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef E02AE628_9C8A_4100_8C73_A3B74B7128F6
#define E02AE628_9C8A_4100_8C73_A3B74B7128F6

#ifdef __cplusplus
extern "C" {
#endif

#define PRINT_INT_LESS_THAN_64(Tliteral, T) void peek_device_data_T##Tliteral(T* d_arr, size_t num, size_t offset);

PRINT_INT_LESS_THAN_64(i8, int8_t)
PRINT_INT_LESS_THAN_64(i16, int16_t)
PRINT_INT_LESS_THAN_64(i32, int32_t)

void peek_device_data_Ti64(int64_t* d_arr, size_t num, size_t offset);

#define PRINT_UINT_LESS_THAN_64(Tliteral, T) void peek_device_data_T##Tliteral(T* d_arr, size_t num, size_t offset);

PRINT_UINT_LESS_THAN_64(ui8, uint8_t)
PRINT_UINT_LESS_THAN_64(ui16, uint16_t)
PRINT_UINT_LESS_THAN_64(ui32, uint32_t)

void peek_device_data_Tui64(uint64_t* d_arr, size_t num, size_t offset);

void peek_device_data_Tfp32(float* d_arr, size_t num, size_t offset);
void peek_device_data_Tfp64(double* d_arr, size_t num, size_t offset);

#undef PRINT_INT_LESS_THAN_64
#undef PRINT_UINT_LESS_THAN_64

#ifdef __cplusplus
}
#endif

#endif /* E02AE628_9C8A_4100_8C73_A3B74B7128F6 */
