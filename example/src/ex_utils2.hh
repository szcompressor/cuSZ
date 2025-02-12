#include <string>
#include <vector>

#include "cusz.h"

struct Arguments {
  std::string fname_prefix;
  std::string fname_suffix;
  int from_number;
  int to_number;
  psz_codectype codec_type{Huffman};
  size_t x;
  size_t y;
  size_t z;
  size_t radius;
};

void print_help();

Arguments parse_arguments(int argc, char* argv[]);

std::vector<std::string> construct_file_names(
    const std::string& prefix, const std::string& suffix, int start, int end, int width);