/**
 * @file rle.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-04-02
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef OOD_RLE_HH
#define OOD_RLE_HH

#include <stdexcept>
#include "../datapack.hh"
#include "../kernel/rle.cuh"

template <typename T>
class RunLengthCodec {
   private:
    DataPack<T>*   fullfmt_data;  // length = N
    DataPack<T>*   compact_data;  // length = N when encoding; length = num_runs when decoding
    DataPack<int>* lengths;       // length = N when encoding; length = num_runs when decoding

    size_t N{}, num_runs{};

   public:
    // variable accessor
    size_t FullLen() const { return N; }
    size_t RunLen() const { return num_runs; }

   public:
    RunLengthCodec(DataPack<T>* _fullfmt_data, DataPack<T>* _compact_data, DataPack<int>* _lengths)
    {
        fullfmt_data = _fullfmt_data;
        compact_data = _compact_data;
        lengths      = _lengths;
    }
    ~RunLengthCodec() = default;

    RunLengthCodec& SetFullLen(size_t _len)
    {
        N = _len;
        return *this;
    }

    RunLengthCodec& SetRunLen(size_t _len)
    {
        num_runs = _len;
        return *this;
    }

    RunLengthCodec& Encode()
    {
        if (fullfmt_data->Len() != N or compact_data->Len() != N or lengths->Len() != N) {
            throw std::runtime_error("For now, full-length and run-length sum must be N.");
        }
        kernel::RunLengthEncoding(fullfmt_data->dptr(), N, compact_data->dptr(), lengths->dptr(), num_runs);
        return *this;
    }

    RunLengthCodec& Decode()
    {
        if (fullfmt_data->Len() != N) { throw std::runtime_error("For now, full-length must be N."); }
        kernel::RunLengthDecoding(fullfmt_data->dptr(), N, compact_data->dptr(), lengths->dptr(), num_runs);
        return *this;
    }
};

#endif
