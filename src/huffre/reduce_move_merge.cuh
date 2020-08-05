// jtian 20-05-30
#include <stdio.h>
#include "auxiliary.cuh"

// https://stackoverflow.com/questions/42309369/can-my-kernel-code-tell-how-much-shared-memory-it-has-available
// usage: if (ti == 0) printf("shared memory size: %u\n", dynamic_smem_size());
__forceinline__ __device__ unsigned dynamic_smem_size()
{
    unsigned ret;
    asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}

template <typename H>
__forceinline__ __device__ int get_bw(H var)
{
    return (int)*((uint8_t*)&var + sizeof(H) - 1);
}

template <int Magnitude>
__forceinline__ __device__ unsigned int ReverseButterflyIdx(unsigned int idx)
{
    auto reversed_idx = 0x0u;
    for (auto i = 0; i < Magnitude; i++) {
        auto tmp = idx & (0x1u << i);
        tmp      = (tmp >> i) << (Magnitude - 1 - i);
        reversed_idx |= tmp;
    }
    return reversed_idx;
}

template <int Magnitude>
__forceinline__ __device__ unsigned int NextButterflyIdx(unsigned int idx, unsigned int iter)
{
    auto team_size        = 1 << (iter - 1);
    auto next_team_size   = 2 * team_size;
    auto league_size      = 1 << (Magnitude - iter + 1);
    auto next_league_size = league_size / 2;

    auto team_rank           = idx % team_size;
    auto league_rank         = idx / team_size;
    auto next_subleague_rank = league_rank / next_league_size;
    auto next_league_rank    = league_rank % next_league_size;
    auto next_rank           = next_league_rank * next_team_size + next_subleague_rank * team_size + team_rank;

    return next_rank;
}

/*
 * Q* q: quantization code
 * H* cb: Huffman codebook
 * TODO (msb) bitwdith (space) Huffman code (lsb)
 * TODO -> ReduceShuffleMerge
 * TODO -> chunksize == 2 * blockDim.x
 */
template <typename Q, typename H, int Magnitude, int ReductionFactor, int ShuffleFactor>
__global__ void ReduceShuffle_Butterfly(Q* q, size_t len, H* cb, H* h, size_t cb_len, uint32_t* hmeta, char* rich_dbg = nullptr, uint32_t dbg_bi = 3)
{
    static_assert(Magnitude == ReductionFactor + ShuffleFactor, "Data magnitude not equal to (shuffle + reduction) factor");
    static_assert(ReductionFactor >= 1, "Reduction factor must be larger than 1");
    static_assert((2 << Magnitude) < 98304, "Shared memory used too much.");

    extern __shared__ char __buff[];

    auto n_worker  = blockDim.x;
    auto chunksize = 1 << Magnitude;
    auto __data    = reinterpret_cast<H*>(__buff);
    auto __bw      = reinterpret_cast<int*>(__buff + (chunksize / 2 + chunksize / 4) * sizeof(H));
    auto data_src  = __data;                  // 1st data zone of (chunksize/2)
    auto data_dst  = __data + chunksize / 2;  // 2nd data zone of (chunksize/4)
    auto data_exc  = data_src;                // swap zone
    auto bw_src    = __bw;                    // 1st bw zone of (chunksize/2)

    //    auto bw_dst    = __bw + chunksize / 2;    // 2nd bw zone of (chunksize/4)
    //    auto bw_exc    = bw_src;                  // swap zone

    auto ti = threadIdx.x;
    auto bi = blockIdx.x;

    //// Reduction I: detach metadata and code; merge as much as possible
    //// butterfly
    ////////////////////////////////////////////////////////////////////////////////
    auto first_stride = 1 << (Magnitude - 1);
    // the same as non-butterfly
    for (auto r = 0; r < (1 << (ReductionFactor - 1)); r++) {  // ReductionFactor - 1 = 2, 8 -> 4 (1 time)
        auto lidx = ti + n_worker * r;                         //
        auto gidx = chunksize * bi + lidx;                     // to load from global memory
        auto lsym = cb[q[gidx]];
        auto rsym = cb[q[gidx + first_stride]];  // the next one
        __syncthreads();

        auto lbw = get_bw(lsym);
        auto rbw = get_bw(rsym);
        lsym <<= sizeof(H) * 8 - lbw;  // left aligned
        rsym <<= sizeof(H) * 8 - rbw;  //

        data_src[lidx] = lsym | (rsym >> lbw);  //
        bw_src[lidx]   = lbw + rbw;             // sum bitwidths
        //        if (bi == dbg_bi) printf("reduce=1; idx=(%u,%u);\tbw=(%u,%u);\n", lidx, lidx + first_stride, lbw, rbw);
    }
    __syncthreads();

#ifdef REDUCE1TIME
    return;
#endif

    //// Reduction II: merge as much as possible
    //// butterfly
    ////////////////////////////////////////////////////////////////////////////////
    auto stride = 1 << (Magnitude - 2);
    for (auto rf = ReductionFactor - 2; rf >= 0; rf--) {  // ReductionFactor - 2 = 1, 4 -> 2 -> 1 (2 times)
        auto repeat = 1 << rf;
        for (auto r = 0; r < repeat; r++) {
            auto lidx = ti + n_worker * r;
            auto lsym = data_src[lidx];
            auto rsym = data_src[lidx + stride];
            auto lbw  = bw_src[lidx];
            auto rbw  = bw_src[lidx + stride];

            data_src[lidx] = lsym | (rsym >> lbw);
            bw_src[lidx]   = lbw + rbw;  // sum bitwidths
            __syncthreads();
            //            if (bi == dbg_bi) printf("rf=%u; idx=(%u,%u)\tbw=(%u,%u)\n", rf, lidx, lidx + stride, lbw, rbw);
        }
        stride /= 2;
        __syncthreads();
    }
    __syncthreads();

#ifdef REDUCE12TIME
    return;
#endif

    //// Shuffle, from now on, 1 thread on 1 point; TODO stride <-> ShuffleFactor
    //// butterfly
    ////////////////////////////////////////////////////////////////////////////////
    auto curr_team_halfsize = 1;
    auto curr_league_size   = 1 << ShuffleFactor;
    auto next_league_size   = curr_league_size / 2;

    for (auto sf = ShuffleFactor; sf > 0; sf--, curr_team_halfsize *= 2, curr_league_size /= 2, next_league_size /= 2) {
        auto rank        = ti;
        auto iter        = ShuffleFactor + 1 - sf;
        auto next_rank   = NextButterflyIdx<ShuffleFactor>(rank, iter);
        auto team_size   = 1 << (iter - 1);
        auto league_size = 1 << (Magnitude - iter + 1);
        auto league_rank = rank / team_size;

        auto l_bw = bw_src[league_rank];
        auto r_bw = bw_src[league_rank + league_size / 2];

        unsigned int dtype_ofst  = l_bw / (sizeof(H) * 8);
        unsigned int used_bits   = l_bw % (sizeof(H) * 8);
        unsigned int unused_bits = sizeof(H) * 8 - used_bits;

        // next team rank can be disrupted from previous snippet
        auto leader = next_rank / (team_size * 2) * (team_size * 2);  // team leader
        auto r      = leader + team_size;
        auto l_end  = data_dst + leader + dtype_ofst;

        auto this_point = data_src[rank];
        /* experimental */ __syncthreads();

        atomicAnd(data_src + rank, 0x0);  // meaning: data_src[ti] = 0;

        if (next_rank >= leader and next_rank < leader + team_size) {
            atomicOr(data_dst + next_rank, this_point);
        }
        if (next_rank >= r and next_rank < r + team_size) {  // whole right subgroup
            auto _1st = this_point >> used_bits;
            auto _2nd = this_point << unused_bits;
            atomicOr(l_end + (next_rank - r) + 0, _1st);  // meaning: *(l_end+(next_rank-r)+0)=_1st;
            atomicOr(l_end + (next_rank - r) + 1, _2nd);  // meaning: *(l_end+(next_rank-rank-r)+0)=_1st;
        }

        if (next_rank == leader) bw_src[league_rank] = l_bw + r_bw;  // very imbalanced

        data_exc = data_src, data_src = data_dst, data_dst = data_exc;
        /* necessary */ __syncthreads();
    }
    /* necessary */ __syncthreads();

    //// end of reduce-shuffle
    ////////////////////////////////////////////////////////////////////////////////
    auto final_bw = bw_src[0];
    if (ti == 0) hmeta[bi] = final_bw;
    __syncthreads();

#ifdef ALLMERGETIME
    return;
#endif

    auto multiple_of_128B = (final_bw - 1) / (sizeof(H) * 8) + 1;
    multiple_of_128B      = ((multiple_of_128B - 1) / 32 + 1) * 32;
    if (ti < multiple_of_128B) h[chunksize * bi + ti] = data_src[ti];
    __syncthreads();
}

/*
 * Q* q: quantization code
 * H* cb: Huffman codebook
 * TODO (msb) bitwdith (space) Huffman code (lsb)
 * TODO -> ReduceShuffleMerge
 * TODO -> chunksize == 2 * blockDim.x
 */
template <typename Q, typename H, int Magnitude, int ReductionFactor, int ShuffleFactor>
__global__ void ReduceShuffle(Q* q, size_t len, H* cb, H* h, size_t cb_len, uint32_t* hmeta, char* rich_dbg = nullptr, uint32_t dbg_bi = 3)
{
    static_assert(Magnitude == ReductionFactor + ShuffleFactor, "Data magnitude not equal to (shuffle + reduction) factor");
    static_assert(ReductionFactor >= 1, "Reduction factor must be larger than 1");
    static_assert((2 << Magnitude) < 98304, "Shared memory used too much.");

    extern __shared__ char __buff[];

    auto n_worker  = blockDim.x;
    auto chunksize = 1 << Magnitude;
    auto __data    = reinterpret_cast<H*>(__buff);
    auto __bw      = reinterpret_cast<int*>(__buff + (chunksize / 2 + chunksize / 4) * sizeof(H));
    auto data_src  = __data;                  // 1st data zone of (chunksize/2)
    auto data_dst  = __data + chunksize / 2;  // 2nd data zone of (chunksize/4)
    auto data_exc  = data_src;                // swap zone
    auto bw_src    = __bw;                    // 1st bw zone of (chunksize/2)
    auto bw_dst    = __bw + chunksize / 2;    // 2nd bw zone of (chunksize/4)
    auto bw_exc    = bw_src;                  // swap zone

    auto ti = threadIdx.x;
    auto bi = blockIdx.x;

    //// Reduction I: detach metadata and code; merge as much as possible
    ////////////////////////////////////////////////////////////////////////////////
    for (auto r = 0; r < (1 << (ReductionFactor - 1)); r++) {  // ReductionFactor - 1 = 2, 8 -> 4 (1 time)
        auto lidx = 2 * (ti + n_worker * r);                   // every two
        auto gidx = chunksize * bi + lidx;                     // to load from global memory

        auto lsym = cb[q[gidx]];
        auto rsym = cb[q[gidx + 1]];  // the next one
        __syncthreads();

        auto lbw = get_bw(lsym);
        auto rbw = get_bw(rsym);
        lsym <<= sizeof(H) * 8 - lbw;  // left aligned
        rsym <<= sizeof(H) * 8 - rbw;  //

        data_src[lidx >> 1] = lsym | (rsym >> lbw);  //
        bw_src[lidx >> 1]   = lbw + rbw;             // sum bitwidths

#ifdef DBG
        if (bi == dbg_bi) {
            printf(
                "blkid=%u; idx=%4u; first_reduce=%u; bw=(%2u, %2u)=%u; sym=(%u, %u); merged=(%u)\n",  //
                dbg_bi, _1st_idx, r, _1st_bw, _2nd_bw, (_1st_bw + _2nd_bw), _1st, _2nd, (_1st | (_2nd >> _1st_bw)));
        }
        __syncthreads();
#endif
    }
    __syncthreads();

#ifdef REDUCE1TIME
    return;
#endif

    //// Reduction II: merge as much as possible
    ////////////////////////////////////////////////////////////////////////////////
    for (auto rf = ReductionFactor - 2; rf >= 0; rf--) {  // ReductionFactor - 2 = 1, 4 -> 2 -> 1 (2 times)
        auto repeat = 1 << rf;
        for (auto r = 0; r < repeat; r++) {
            auto lidx = 2 * (ti + n_worker * r);
            auto lsym = data_src[lidx];
            auto rsym = data_src[lidx + 1];
            auto lbw  = bw_src[lidx];
            auto rbw  = bw_src[lidx + 1];
#ifdef DBG
            if (bi == dbg_bi)
                printf(
                    "blkid=%u; idx=%4u; loop=%u; repeat=%u; bw=(%2u, %2u)=%2u; sym=(%u, %u); merged=(%u)\n",  //
                    dbg_bi, lidx, rf, r, lbw, rbw, (lbw + rbw), lsym, rsym, (lsym | (rsym >> lbw)));
#endif

            data_dst[lidx >> 1] = lsym | (rsym >> lbw);
            bw_dst[lidx >> 1]   = lbw + rbw;  // sum bitwidths
            __syncthreads();
        }
        __syncthreads();
        data_exc = data_src, bw_exc = bw_src;
        data_src = data_dst, bw_src = bw_dst;
        data_dst = data_exc, bw_dst = bw_exc;
        __syncthreads();
    }
    __syncthreads();

#ifdef REDUCE12TIME
    return;
#endif

#ifdef DBG
    if (bi == dbg_bi and ti == 0) {
        for (auto ii = 0; ii < (1 << ShuffleFactor); ii++)
            if (data_src[ii] != 0x0) printf("idx=%3u;\tbw=%3u; code=0b", ii, bw_src[ii]), echo_bitset<H, 32>(data_src[ii]);
        printf("\n");
    }
    __syncthreads();
#endif

    //// Shuffle, from now on, 1 thread on 1 point; TODO stride <-> ShuffleFactor
    ////////////////////////////////////////////////////////////////////////////////
    auto stride = 1;
    for (auto sf = ShuffleFactor; sf > 0; sf--, stride *= 2) {
        auto         l           = ti / (stride << 1) * (stride << 1);
        auto         r           = l + stride;
        auto         l_bw        = bw_src[l];
        unsigned int dtype_ofst  = l_bw / (sizeof(H) * 8);
        unsigned int used_bits   = l_bw % (sizeof(H) * 8);
        unsigned int unused_bits = sizeof(H) * 8 - used_bits;
        auto         l_end       = data_src + l + dtype_ofst;

        auto this_point = data_src[ti];
        auto _1st       = this_point >> used_bits;
        auto _2nd       = this_point << unused_bits;

#ifdef dgb
        if (bi == dbg_bi and ti == 0)
            for (auto ii = 0; ii < (1 << ShuffleFactor); ii++) {
                printf("sf_iter=%u; data_idx=%u\t", sf, ii);
                if (data_src[ii] != 0x0)
                    printf("bw=%3u; code=", bw_src[ii]), echo_bitset<H, 32>(data_src[ii]);
                else
                    printf("bw=nan; code="), printf("0x0                             \n");
            }
        __syncthreads();
#endif

        // ti in [r, r+stride) or ((r ..< r+stride )), whole right subgroup
        // &data_src[ ((r ..< r+stride)) ] have conflicts with (l_end+ ((0 ..< stride)) + 0/1)
        // because the whole shuffle subprocedure compresses at a factor of < 2
        if (ti >= r and ti < r + stride) {
            atomicAnd(data_src + ti, 0x0);  // meaning: data_src[ti] = 0;
        }
        /* experimental */ __syncthreads();
        if (ti >= r and ti < r + stride) {         // whole right subgroup
            atomicOr(l_end + (ti - r) + 0, _1st);  // meaning: *(l_end + (ti -r) + 0) = _1st;
            atomicOr(l_end + (ti - r) + 1, _2nd);  // meaning: *(l_end + (ti -r) + 0) = _1st;
        }
        ///* optional */ __syncthreads();

        if (ti == l) bw_src[l] += bw_src[l + stride];  // very imbalanced
        /* necessary */ __syncthreads();
    }
    /* necessary */ __syncthreads();

    //// end of reduce-shuffle
    ////////////////////////////////////////////////////////////////////////////////
    auto final_bw = bw_src[0];
    if (ti == 0) hmeta[bi] = final_bw;
    __syncthreads();

#ifdef ALLMERGETIME
    return;
#endif

    auto multiple_of_128B = (final_bw - 1) / (sizeof(H) * 8) + 1;
    multiple_of_128B      = ((multiple_of_128B - 1) / 32 + 1) * 32;
    if (ti < multiple_of_128B) h[chunksize * bi + ti] = data_src[ti];
    __syncthreads();

    /*
    if (bi == dbg_bi and ti == 0) {
        printf("dbg blk id:%u (printing)\n", dbg_bi);
        for (auto i = 0; i < multiple_of_128B; i++) echo_bitset<uint32_t, 32>(h[chunksize * bi + i]);
    }
    */

    // we can
    // g.sync();
    // prefix sum

    // copy dense Huffman codes to global memory
    // auto n_of_128 = (final_bw / 8 - 1) / 128 + 1; // if we go warp ops
}
// |_0___________||_1___________||_2___________|
// |xxxxx________||xxxxxxxx_____||xxxxxx_______|
//  ^              ^
//  |              |
//  |    ^
//  | #0 | #1
//  2044 (4)       3070 (2)       2040 (8)
//  16             23.7 -> 24     15.7
// dense_h array: storing huffman bitstream
