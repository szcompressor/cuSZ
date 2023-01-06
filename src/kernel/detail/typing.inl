/**
 * @file typing.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

// TODO -> `compaction`
template <typename T>
struct CompactionDRAM {
    using type = T;
    T*        val;
    uint32_t* idx;
    uint32_t* count;

    void allocate(size_t len, bool device = true)
    {
        if (device) {
            cudaMalloc(&idx, sizeof(uint32_t) * len);
            cudaMalloc(&val, sizeof(T) * len);
            cudaMalloc(&count, sizeof(T) * 1);

            cudaMemset(count, 0x0, sizeof(T) * 1);
        }
        else {
            cudaMallocHost(&idx, sizeof(uint32_t) * len);
            cudaMallocHost(&val, sizeof(T) * len);
            cudaMallocHost(&count, sizeof(T) * 1);

            memset(count, 0x0, sizeof(T) * 1);
        }
    }

    void allocate_managed(size_t len)
    {
        cudaMallocManaged(&idx, sizeof(uint32_t) * len);
        cudaMallocManaged(&val, sizeof(T) * len);
        cudaMallocManaged(&count, sizeof(T) * 1);

        cudaMemset(count, 0x0, sizeof(T) * 1);
    }

    void destroy()
    {
        cudaFree(idx);
        cudaFree(val);
        cudaFree(count);
    }
};
