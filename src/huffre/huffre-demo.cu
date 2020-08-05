//
// Created by jtian on 4/24/20.
//

#include <cuda_runtime.h>

#include <sys/stat.h>
#include <bitset>
#include <cassert>
#include <iostream>
#include <string>
#include <tuple>

#include <stdio.h>

#include "../cuda_error_handling.cuh"
#include "../cuda_mem.cuh"
#include "../huffman_codec.cuh"
#include "../huffman_workflow.cuh"
#include "../timer.hh"
#include "../types.hh"

#include "reduce_move_merge.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

uint32_t dbg_bi = 2;

template <typename H>
inline int get_symlen(H sym)
{
    return (int)*((uint8_t*)&sym + sizeof(H) - 1);
}

template <typename Q, typename H>
void synthesize_q(Q* q, uint32_t len, H* cb)  // prepare for extra outliers
{
    for (auto i = 0; i < len; i++) {
        auto sym     = cb[q[i]];
        auto sym_len = get_symlen(sym);
        if (sym_len > 5) q[i] = 0;
    }
}

template <typename Q, typename H, int Magnitude, int ReductionFactor, int ShuffleFactor>
std::tuple<uint32_t, H*, uint32_t*>
new_enc(Q* q, uint32_t len, H* cb, uint32_t cb_len, /*H* h, uint32_t* hmeta,*/ uint32_t dummy_nchunk_, bool use_dummylen = false)
{
    // TODO auto d_rich_dbg = mem::CreateCUDASpace<char>(len * 4);
    auto d_q       = mem::CreateDeviceSpaceAndMemcpyFromHost(q, len);
    auto d_cb      = mem::CreateDeviceSpaceAndMemcpyFromHost(cb, cb_len);
    auto d_h       = mem::CreateCUDASpace<H>(len);
    auto chunksize = 1 << Magnitude;
    if (use_dummylen) len = dummy_nchunk_ * chunksize;
    auto blockDim   = 1 << ShuffleFactor;
    auto gridDim    = len / chunksize;
    auto buff_bytes = (chunksize / 2 + chunksize / 4) * (sizeof(H) + sizeof(int));
    // share memory usage: 1.5 * chunksize * 4 = 6 * chunksize: 6 * 1K = 6K
    // data size: sizeof(uint16_t) * chunksize: 2 * 1024
    // thread number : chunksize >> 3, 128, at max 2* 1024 / 128 = 16 threadblocks on 1 SM
    cout << "len     \t" << len << endl;
    cout << "chunksize\t" << chunksize << endl;
    cout << "blockDim\t" << blockDim << endl;
    cout << "gridDim \t" << gridDim << endl;
    cout << "per-block shmem bytes\t" << buff_bytes << "\t" << 96 * 1024 / buff_bytes << " blocks EXPECTED on 1 SM" << endl;
    cout << 1024 * 2 / blockDim << " should be blocks on 1 SM" << endl;
    auto d_hmeta = mem::CreateCUDASpace<uint32_t>(gridDim);         // chunkwise metadata
    ReduceShuffle<Q, H, Magnitude, ReductionFactor, ShuffleFactor>  //
        <<<gridDim, blockDim, buff_bytes>>>(d_q, len, d_cb, d_h, cb_len, d_hmeta, nullptr, dbg_bi);
    HANDLE_ERROR(cudaDeviceSynchronize());
    auto h     = mem::CreateHostSpaceAndMemcpyFromDevice(d_h, len);
    auto hmeta = mem::CreateHostSpaceAndMemcpyFromDevice(d_hmeta, gridDim);
    cudaFree(d_q), cudaFree(d_cb), cudaFree(d_h), cudaFree(d_hmeta);
    cout << "new enc gracefully quited." << endl;
    return {gridDim, h, hmeta};
}

template <typename Q, typename H, int Magnitude>
std::tuple<uint32_t, H*, size_t*>
old_enc(Q* q, uint32_t len, H* cb, uint32_t cb_len, /*H* h, size_t* hmeta, */ uint32_t dummy_nchunk_, bool use_dummylen = false)
{
    auto chunksize = 1 << Magnitude;
    if (use_dummylen) len = dummy_nchunk_ * chunksize;
    auto n_chunk = (len - 1) / chunksize + 1;
    // TODO auto d_rich_dbg = mem::CreateCUDASpace<char>(len * 4);
    auto d_q     = mem::CreateDeviceSpaceAndMemcpyFromHost(q, len);
    auto d_cb    = mem::CreateDeviceSpaceAndMemcpyFromHost(cb, cb_len);
    auto d_h     = mem::CreateCUDASpace<H>(len);
    auto d_hmeta = mem::CreateCUDASpace<size_t>(n_chunk);
    {
        auto blockDim = 128;
        auto gridDim  = (len - 1) / blockDim + 1;
        EncodeFixedLen<Q, H><<<gridDim, blockDim>>>(d_q, d_h, len, d_cb);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
    {
        auto blockDim = 128;
        auto gridDim  = (n_chunk - 1) / blockDim + 1;
        Deflate<H><<<gridDim, blockDim>>>(d_h, len, d_hmeta, chunksize);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
    auto h     = mem::CreateHostSpaceAndMemcpyFromDevice(d_h, len);
    auto hmeta = mem::CreateHostSpaceAndMemcpyFromDevice(d_hmeta, n_chunk);
    cudaFree(d_q), cudaFree(d_cb), cudaFree(d_h), cudaFree(d_hmeta);
    cout << "old enc gracefully quited." << endl;
    return {n_chunk, h, hmeta};
}

template <typename H>
void check_afterward(
    std::tuple<uint32_t, H*, uint32_t*> ne,
    std::tuple<uint32_t, H*, size_t*>   oe,
    unsigned int                        len,
    double                              entropy,
    unsigned int                        ChunkSize,
    unsigned int                        override_blks_check = 0)
{
    // depack results
    auto ne_nchunk = std::get<0>(ne);  //
    auto ne_h      = std::get<1>(ne);  //
    auto ne_hmeta  = std::get<2>(ne);  //
    auto oe_nchunk = std::get<0>(oe);  //
    auto oe_h      = std::get<1>(oe);  //
    auto oe_hmeta  = std::get<2>(oe);  //

    if (ne_nchunk != oe_nchunk) {
        cerr << "new encoder nchunk != old enc nchunk" << endl;
        exit(1);
    }

    auto all_meta_equal = true;
    for (auto i = 0; i < ne_nchunk; i++)
        if (ne_hmeta[i] != oe_hmeta[i]) cout << "chunk " << i << ": ne_hmeta != oe_hmeta" << endl, all_meta_equal = false;

    if (all_meta_equal) cout << "all meta equal for ne and oe" << endl;

    auto count_bad_chunks = 0, count_bad_uint32 = 0;
    auto num_check_blk = ne_nchunk;
    num_check_blk      = override_blks_check == 0 ? ne_nchunk : override_blks_check;

    // for (auto i = 1070; i < 1080; i++) {
    for (auto i = 0, count = 0; i < ne_nchunk; i++) {
        for (auto j = 0; j < (ne_hmeta[i] - 1) / (sizeof(H) * 8) + 1; j++) {
            auto idx = ChunkSize * i + j;
            if (ne_h[idx] != oe_h[idx]) {
                cout << "hmeta=" << ne_hmeta[i] << " block " << i << " dtype " << j << endl;
                cout << bitset<32>(ne_h[idx]) << "\t(new enc)\n" << bitset<32>(oe_h[idx]) << "\t(old enc)" << endl;
                for (auto ii = 0; ii < sizeof(H) * 8; ii++) {
                    auto ne_bit = (ne_h[idx] >> (sizeof(H) * 8 - 1 - ii)) & 0x1u;
                    auto oe_bit = (oe_h[idx] >> (sizeof(H) * 8 - 1 - ii)) & 0x1u;
                    cout << (ne_bit != oe_bit ? "^" : " ");
                }
                cout << endl, count++;
            }
        }
        if (count != 0) {
            cout << count << " dtypes are affected." << endl;
            count_bad_uint32 += count, count_bad_chunks++;
        }
    }
    cout << "# bad chunks: " << count_bad_chunks << " out of " << num_check_blk << endl;
    cout << "# bad uint32: " << count_bad_uint32 << " out of " << static_cast<int>(len * entropy / 32) << endl;

    delete[] ne_h, ne_hmeta, oe_h, oe_hmeta;
}

int main(int argc, char** argv)
{
    typedef uint16_t Qtype;
    typedef uint32_t Htype;

    string   f_q, f_cb;
    uint32_t dummy_nchunk = 0;
    if (argc < 3) {
        f_q = string("baryon_density.dat.b16"), f_cb = string("baryon_density.dat.b16.canonized");
        cout << "./huff <quant code> <codebook> <dummy #chunk>\nusing default: " << f_q << "\t" << f_cb << endl;
        dummy_nchunk = 0;
    }
    else {
        f_q = string(argv[1]), f_cb = string(argv[2]);
        dummy_nchunk = atoi(argv[3]);
    }
    cudaDeviceReset();

    auto cb_len = 1024, len = 512 * 512 * 512;
    auto q  = io::ReadBinaryFile<Qtype>(f_q, len);
    auto cb = io::ReadBinaryFile<Htype>(f_cb, cb_len);

    auto symbol_zero = cb[0];
    cout << "symbol zero len: " << get_symlen(symbol_zero) << "\t" << bitset<32>(symbol_zero) << endl;

    synthesize_q(q, len, cb);  // prepare for extra outliers

    //    for (auto i = 0; i < 200; i++) cout << i << "\t" << bitset<16>(q[i]) << endl;      // check q   [PASS]
    //    for (auto i = 0; i < cb_len; i++) cout << i << "\t" << bitset<32>(cb[i]) << endl;  // check cb  [PASS]

    // TODO histogram -> entropy
    double entropy;
    {
        auto d_q_hist = mem::CreateDeviceSpaceAndMemcpyFromHost(q, len);
        auto d_freq   = mem::CreateCUDASpace<unsigned int>(cb_len);
        wrapper::GetFrequency(d_q_hist, len, d_freq, cb_len);
        auto freq = mem::CreateHostSpaceAndMemcpyFromDevice(d_freq, cb_len);
        entropy   = 0;
        for (auto i = 0; i < cb_len; i++) {
            auto bw = get_symlen(cb[i]);
            if (bw != 255) entropy += static_cast<double>(freq[i]) * static_cast<double>(bw);
        }
        entropy /= len;
        cout << "entropy: " << entropy << endl;
        cudaFree(d_q_hist), cudaFree(d_freq);
    }

    dbg_bi               = 0;   // debug only
    const auto Magnitude = 10;  // 1 << 10, 1024 point per chunk
    const auto ChunkSize = 1 << Magnitude;

    std::tuple<uint32_t, Htype*, uint32_t*> ne;
    std::tuple<uint32_t, Htype*, size_t*>   oe;

    if (entropy >= 2 and entropy < 4) {
        const auto ReductionFactor = 3;  // (2.0 to <4.0) * 8 = (16.0 to <32.0); 1 << (10-3) == 128
        const auto ShuffleFactor   = Magnitude - ReductionFactor;
        ne                         = new_enc<Qtype, Htype, Magnitude, ReductionFactor, ShuffleFactor>  // new encoder
            (q, len, cb, cb_len, dummy_nchunk, dummy_nchunk != 0);                                     //
        oe = old_enc<Qtype, Htype, Magnitude>                                                          // old encoder
            (q, len, cb, cb_len, dummy_nchunk, dummy_nchunk != 0);                                     //
    }
    else if (entropy >= 4 and entropy < 8) {
        cout << "not yet" << endl;
        const auto ReductionFactor = 2;  // (2.0 to <4.0) * 8 = (16.0 to <32.0); 1 << (10-3) == 128
        const auto ShuffleFactor   = Magnitude - ReductionFactor;

        ne = new_enc<Qtype, Htype, Magnitude, ReductionFactor, ShuffleFactor>  // new encoder
            (q, len, cb, cb_len, dummy_nchunk, dummy_nchunk != 0);             //
        oe = old_enc<Qtype, Htype, Magnitude>                                  // old encoder
            (q, len, cb, cb_len, dummy_nchunk, dummy_nchunk != 0);             //
    }

#ifndef REDUCE1TIME
#ifndef REDUCE12TIME
#ifndef ALLMERGETIME
    check_afterward(ne, oe, len, entropy, ChunkSize);
#endif
#endif
#endif
    delete[] q, cb;
    return 0;
}
