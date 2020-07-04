//
// By Cody Rivera, 7/2020
//

// Sorts quantization codes by frequency, using a key-value
// sort.

// I have placed this functionality in a separate compilation
// unit as thrust calls fail in par_huffman.cu.

#include <thrust/device_vector.h>
#include <thrust/sort.h>

void SortByFreq(unsigned int* freq, int* qcode, int size) {
    using namespace thrust;
    sort_by_key(device_ptr<unsigned int>(freq),
                device_ptr<unsigned int>(freq + size),
                device_ptr<int>(qcode));
}