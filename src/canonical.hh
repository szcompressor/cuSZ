#ifndef CANONICAL_HH
#define CANONICAL_HH

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>

using std::cout;
using std::endl;
using std::for_each;

namespace CPU {

template <typename Q, typename T>
void inflate(Q* densely, T* bcode, size_t total_bw, uint8_t* singleton) {
    uint8_t   next_bit;
    size_t    idx_bit;
    size_t    idx_byte  = 0;
    size_t    idx_bcode = 0;
    auto      first     = reinterpret_cast<int*>(singleton);
    auto      entry     = first + sizeof(Q) * 8;
    auto      keys      = reinterpret_cast<uint16_t*>(singleton + sizeof(int) * (2 * sizeof(Q) * 8));
    ptrdiff_t v         = (densely[idx_byte] >> (sizeof(Q) * 8 - 1)) & 0x1;  // get the first bit
    size_t    l         = 1;
    size_t    i         = 0;
    while (i < total_bw) {
        while (v < first[l]) {  // append next input bit
            ++i;
            idx_byte = i / (sizeof(Q) * 8);
            idx_bit  = i % (sizeof(Q) * 8);
            next_bit = ((densely[idx_byte] >> (sizeof(Q) * 8 - 1 - idx_bit)) & 0x1);
            v        = (v << 1) | next_bit;
            l++;
        }
        bcode[idx_bcode++] = keys[entry[l] + v - first[l]];
        {
            ++i;
            idx_byte = i / (sizeof(Q) * 8);
            idx_bit  = i % (sizeof(Q) * 8);
            next_bit = ((densely[idx_byte] >> (sizeof(Q) * 8 - 1 - idx_bit)) & 0x1);
            v        = 0x0 | next_bit;
        }
        l = 1;
    }
}

template <typename Q>
void deflate_v3(Q* hcode, size_t len, size_t* metadata_deflated) {
    uint8_t bw;
    size_t  lsb_pos = sizeof(Q) * 8, total_bw = 0;

    size_t ending = len;
    Q      msb_bw__hcode_lsb, _1, _2;
    Q*     cur = hcode;
    for (size_t i = 0; i < ending; i++) {
        msb_bw__hcode_lsb = hcode[i];
        bw                = *((uint8_t*)&msb_bw__hcode_lsb + (sizeof(Q) - 1));

        *((uint8_t*)&msb_bw__hcode_lsb + sizeof(Q) - 1) = 0x0;
        if (lsb_pos == sizeof(Q) * 8) *cur = 0x0;  // a new unit of data type
        if (bw <= lsb_pos) {
            lsb_pos -= bw;
            *cur |= msb_bw__hcode_lsb << lsb_pos;
            if (lsb_pos == 0) {
                lsb_pos = sizeof(Q) * 8;
                ++cur;
            }
        } else {
            _1 = msb_bw__hcode_lsb >> (bw - lsb_pos);
            _2 = msb_bw__hcode_lsb << (sizeof(Q) * 8 - (bw - lsb_pos));
            *cur |= _1;
            *(++cur) = 0x0;
            *cur |= _2;
            lsb_pos = sizeof(Q) * 8 - (bw - lsb_pos);
        }
        total_bw += bw;
    }
    *metadata_deflated = total_bw;
}

template <typename T, typename K>
uint8_t GetCanonicalCode(uint8_t* singleton, size_t len) {
    auto type_bw   = sizeof(T) * 8;
    auto codebooks = reinterpret_cast<T*>(singleton);
    auto metadata  = reinterpret_cast<int*>(singleton + sizeof(T) * (3 * len));
    auto keys      = reinterpret_cast<K*>(singleton + sizeof(T) * (3 * len) + sizeof(int) * (4 * type_bw));
    T*   i_cb      = codebooks;
    T*   o_cb      = codebooks + len;
    T*   canonical = codebooks + len * 2;
    auto numl      = metadata;
    auto iter_by_  = metadata + type_bw;
    auto first     = metadata + type_bw * 2;
    auto entry     = metadata + type_bw * 3;

    uint8_t max_bw = 0;

    for_each(i_cb, i_cb + len, [&](T& c) {
        if (c != ~((T)0x0) and c != 0x0) {
            uint8_t bw = *((uint8_t*)&c + (sizeof(T) - 1));
            max_bw     = max_bw < bw ? bw : max_bw;
            numl[bw]   = numl[bw] + 1;
        }
    });

    std::memcpy(entry + 1, numl, sizeof(int) * (type_bw - 1));
    for (size_t i = 1; i < type_bw; i++) entry[i] += entry[i - 1];

    std::memcpy(iter_by_, entry, sizeof(int) * type_bw);

    //    cout << "l=" << max_bw << ;
    for (ptrdiff_t l = max_bw - 1; l >= 1; l--) {
        first[l] = static_cast<int>((first[l + 1] + numl[l + 1]) / 2.0 + 0.5);
        //        cout << ceil()
    }
    first[0] = 0xff;  // no off-by-one error

    std::memset(canonical, 0xff, sizeof(T) * len);
    std::memset(o_cb, 0xff, sizeof(T) * len);

    for (size_t i = 0; i < len; i++) {
        auto    c  = i_cb[i];
        uint8_t bw = *((uint8_t*)&c + (sizeof(T) - 1));
        if (c == ~((T)0x0)) continue;

        canonical[iter_by_[bw]] = static_cast<T>(first[bw] + iter_by_[bw] - entry[bw]);
        keys[iter_by_[bw]]      = i;

        *((uint8_t*)&canonical[iter_by_[bw]] + sizeof(T) - 1) = bw;
        iter_by_[bw]++;
    }

    for (size_t i = 0; i < len; i++) {
        if (canonical[i] == ~((T)0x0u)) continue;
        o_cb[keys[i]] = canonical[i];

#ifdef DEBUG
        cout << "canonical code example, sorted by bitwidth:" << endl;
        for (size_t i = 0; i < len; i++) {
            auto    c  = canonical[i];
            uint8_t bw = *((uint8_t*)&c + (sizeof(T) - 1));

            if (c != ~((T)0x0) and i < 20) {
                cout << "key: " << keys[i] << "\tcode: " << std::bitset<sizeof(T) * 8>(c) << endl;
            }
        }
        cout << "..." << endl << endl;
        cout << "printing all the codebook that will be in use:" << endl;
        for (size_t i = 0; i < len; i++) {
            if (o_cb[i] == ~((T)0x0)) continue;
            cout << "idx: " << i << "\tcode: " << std::bitset<sizeof(T) * 8>(o_cb[i]) << endl;
        }
#endif

        return max_bw;
    }

}  // namespace CPU

#endif
