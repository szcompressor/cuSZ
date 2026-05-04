
#define NTH_BIT(N) (0x01 & (n >> (sizeof(T) * 8 - N)))

template <typename T, int WIDTH>
__forceinline__ __device__ void echo_bitset(T n)
{
  if (WIDTH == 2)
    printf(
        "%d%d\n",                     //
        NTH_BIT(0x01), NTH_BIT(0x02)  //
    );
  if (WIDTH == 3)
    printf(
        "%d%d%d\n",                                  //
        NTH_BIT(0x01), NTH_BIT(0x02), NTH_BIT(0x03)  //
    );
  if (WIDTH == 4)
    printf(
        "%d%d%d%d\n",                                               //
        NTH_BIT(0x01), NTH_BIT(0x02), NTH_BIT(0x03), NTH_BIT(0x04)  //
    );
  if (WIDTH == 8)
    printf(
        "%d%d%d%d%d%d%d%d\n",                                        //
        NTH_BIT(0x01), NTH_BIT(0x02), NTH_BIT(0x03), NTH_BIT(0x04),  //
        NTH_BIT(0x05), NTH_BIT(0x06), NTH_BIT(0x07), NTH_BIT(0x08)   //
    );
  if (WIDTH == 16)
    printf(
        "%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n",                        //
        NTH_BIT(0x01), NTH_BIT(0x02), NTH_BIT(0x03), NTH_BIT(0x04),  //
        NTH_BIT(0x05), NTH_BIT(0x06), NTH_BIT(0x07), NTH_BIT(0x08),  //
        NTH_BIT(0x09), NTH_BIT(0x0a), NTH_BIT(0x0b), NTH_BIT(0x0c),  //
        NTH_BIT(0x0d), NTH_BIT(0x0e), NTH_BIT(0x0f), NTH_BIT(0x10)   //
    );
  if (WIDTH == 32)
    printf(
        "%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d"
        "%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n",                        //
        NTH_BIT(0x01), NTH_BIT(0x02), NTH_BIT(0x03), NTH_BIT(0x04),  //
        NTH_BIT(0x05), NTH_BIT(0x06), NTH_BIT(0x07), NTH_BIT(0x08),  //
        NTH_BIT(0x09), NTH_BIT(0x0a), NTH_BIT(0x0b), NTH_BIT(0x0c),  //
        NTH_BIT(0x0d), NTH_BIT(0x0e), NTH_BIT(0x0f), NTH_BIT(0x10),  //
        NTH_BIT(0x11), NTH_BIT(0x12), NTH_BIT(0x13), NTH_BIT(0x14),  //
        NTH_BIT(0x15), NTH_BIT(0x16), NTH_BIT(0x17), NTH_BIT(0x18),  //
        NTH_BIT(0x19), NTH_BIT(0x1a), NTH_BIT(0x1b), NTH_BIT(0x1c),  //
        NTH_BIT(0x1d), NTH_BIT(0x1e), NTH_BIT(0x1f), NTH_BIT(0x20)   //
    );
}
