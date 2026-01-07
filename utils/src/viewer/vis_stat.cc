#include "utils/vis_stat.hh"

#include "vis_stat.inl"

template double get_entropy<uint8_t>(uint8_t* code, size_t l, size_t cap);
template double get_entropy<uint16_t>(uint16_t* code, size_t l, size_t cap);
template double get_entropy<uint32_t>(uint32_t* code, size_t l, size_t cap);

template void visualize_histogram<uint8_t>(
    const std::string&, uint8_t*, size_t, size_t, bool, double, double, bool, bool);
template void visualize_histogram<uint16_t>(
    const std::string&, uint16_t*, size_t, size_t, bool, double, double, bool, bool);
template void visualize_histogram<uint32_t>(
    const std::string&, uint32_t*, size_t, size_t, bool, double, double, bool, bool);
