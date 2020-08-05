
template <typename T, int width>
__forceinline__ __device__ void echo_bitset(T n)
{
    if (width == 2)
        printf(
            "%d%d\n",                           //
            0x01 & (n >> (sizeof(T) * 8 - 1)),  //
            0x01 & (n >> (sizeof(T) * 8 - 2))   //
        );
    if (width == 3)
        printf(
            "%d%d%d\n",                         //
            0x01 & (n >> (sizeof(T) * 8 - 1)),  //
            0x01 & (n >> (sizeof(T) * 8 - 2)),  //
            0x01 & (n >> (sizeof(T) * 8 - 3))   //
        );
    if (width == 4)
        printf(
            "%d%d%d%d\n",                       //
            0x01 & (n >> (sizeof(T) * 8 - 1)),  //
            0x01 & (n >> (sizeof(T) * 8 - 2)),  //
            0x01 & (n >> (sizeof(T) * 8 - 3)),  //
            0x01 & (n >> (sizeof(T) * 8 - 4))   //
        );
    if (width == 8)
        printf(
            "%d%d%d%d%d%d%d%d\n",               //
            0x01 & (n >> (sizeof(T) * 8 - 1)),  //
            0x01 & (n >> (sizeof(T) * 8 - 2)),  //
            0x01 & (n >> (sizeof(T) * 8 - 3)),  //
            0x01 & (n >> (sizeof(T) * 8 - 4)),  //
            0x01 & (n >> (sizeof(T) * 8 - 5)),  //
            0x01 & (n >> (sizeof(T) * 8 - 6)),  //
            0x01 & (n >> (sizeof(T) * 8 - 7)),  //
            0x01 & (n >> (sizeof(T) * 8 - 8))   //
        );
    if (width == 16)
        printf(
            "%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n",  //
            0x01 & (n >> (sizeof(T) * 8 - 1)),     //
            0x01 & (n >> (sizeof(T) * 8 - 2)),     //
            0x01 & (n >> (sizeof(T) * 8 - 3)),     //
            0x01 & (n >> (sizeof(T) * 8 - 4)),     //
            0x01 & (n >> (sizeof(T) * 8 - 5)),     //
            0x01 & (n >> (sizeof(T) * 8 - 6)),     //
            0x01 & (n >> (sizeof(T) * 8 - 7)),     //
            0x01 & (n >> (sizeof(T) * 8 - 8)),     //
            0x01 & (n >> (sizeof(T) * 8 - 9)),     //
            0x01 & (n >> (sizeof(T) * 8 - 10)),    //
            0x01 & (n >> (sizeof(T) * 8 - 11)),    //
            0x01 & (n >> (sizeof(T) * 8 - 12)),    //
            0x01 & (n >> (sizeof(T) * 8 - 13)),    //
            0x01 & (n >> (sizeof(T) * 8 - 14)),    //
            0x01 & (n >> (sizeof(T) * 8 - 15)),    //
            0x01 & (n >> (sizeof(T) * 8 - 16))     //
        );
    if (width == 32)
        printf(
            "%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d"
            "%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n",  //
            0x01 & (n >> (sizeof(T) * 8 - 1)),     //
            0x01 & (n >> (sizeof(T) * 8 - 2)),     //
            0x01 & (n >> (sizeof(T) * 8 - 3)),     //
            0x01 & (n >> (sizeof(T) * 8 - 4)),     //
            0x01 & (n >> (sizeof(T) * 8 - 5)),     //
            0x01 & (n >> (sizeof(T) * 8 - 6)),     //
            0x01 & (n >> (sizeof(T) * 8 - 7)),     //
            0x01 & (n >> (sizeof(T) * 8 - 8)),     //
            0x01 & (n >> (sizeof(T) * 8 - 9)),     //
            0x01 & (n >> (sizeof(T) * 8 - 10)),    //
            0x01 & (n >> (sizeof(T) * 8 - 11)),    //
            0x01 & (n >> (sizeof(T) * 8 - 12)),    //
            0x01 & (n >> (sizeof(T) * 8 - 13)),    //
            0x01 & (n >> (sizeof(T) * 8 - 14)),    //
            0x01 & (n >> (sizeof(T) * 8 - 15)),    //
            0x01 & (n >> (sizeof(T) * 8 - 16)),    //
            0x01 & (n >> (sizeof(T) * 8 - 17)),    //
            0x01 & (n >> (sizeof(T) * 8 - 18)),    //
            0x01 & (n >> (sizeof(T) * 8 - 19)),    //
            0x01 & (n >> (sizeof(T) * 8 - 20)),    //
            0x01 & (n >> (sizeof(T) * 8 - 21)),    //
            0x01 & (n >> (sizeof(T) * 8 - 22)),    //
            0x01 & (n >> (sizeof(T) * 8 - 23)),    //
            0x01 & (n >> (sizeof(T) * 8 - 24)),    //
            0x01 & (n >> (sizeof(T) * 8 - 25)),    //
            0x01 & (n >> (sizeof(T) * 8 - 26)),    //
            0x01 & (n >> (sizeof(T) * 8 - 27)),    //
            0x01 & (n >> (sizeof(T) * 8 - 28)),    //
            0x01 & (n >> (sizeof(T) * 8 - 29)),    //
            0x01 & (n >> (sizeof(T) * 8 - 30)),    //
            0x01 & (n >> (sizeof(T) * 8 - 31)),    //
            0x01 & (n >> (sizeof(T) * 8 - 32))     //
        );
}
