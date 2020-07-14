//
// By Cody Rivera, 7/2020
//

// Sorts quantization codes by frequency, using a key-value
// sort.

// I have placed this functionality in a separate compilation
// unit as thrust calls fail in par_huffman.cu.

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdint>

template <typename K, typename V>
void SortByFreq(K* freq, V* qcode, int size) {
    using namespace thrust;
    sort_by_key(device_ptr<K>(freq),
                device_ptr<K>(freq + size),
                device_ptr<V>(qcode));
}

template void SortByFreq<unsigned int, uint8_t>(unsigned int*, uint8_t*, int);
template void SortByFreq<unsigned int, uint16_t>(unsigned int*, uint16_t*, int);
template void SortByFreq<unsigned int, uint32_t>(unsigned int*, uint32_t*, int);
