#include "ex_utils2.hh"

#include <iomanip>
#include <iostream>
#include <vector>

void print_help()
{
  std::cout << "usage: PROG  [options]\n"
            << "options:\n"
            << "  --fname-prefix PREFIX   file name prefix\n"
            << "  --fname-suffix SUFFIX   file name suffix\n"
            << "  --from NUMBER           start number\n"
            << "  --to NUMBER             end number\n"
            << "  -x NUMBER               dim x\n"
            << "  -y NUMBER               dim y\n"
            << "  -z NUMBER               dim z\n"
            << "  -r NUMBER               radius\n"
            << "  --help                  this message\n";
}

Arguments parse_arguments(int argc, char* argv[])
{
  Arguments args;
  bool fname_prefix_set = false;
  bool fname_suffix_set = false;
  bool from_number_set = false;
  bool to_number_set = false;
  bool x_set = false;
  bool y_set = false;
  bool z_set = false;
  bool radius_set = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--fname-prefix" and i + 1 < argc) {
      args.fname_prefix = argv[++i];
      fname_prefix_set = true;
    }
    else if (arg == "--fname-suffix" and i + 1 < argc) {
      args.fname_suffix = argv[++i];
      fname_suffix_set = true;
    }
    else if (arg == "--from" and i + 1 < argc) {
      args.from_number = atoi(argv[++i]);
      from_number_set = true;
    }
    else if (arg == "--to" and i + 1 < argc) {
      args.to_number = atoi(argv[++i]);
      to_number_set = true;
    }
    else if (arg == "-x" and i + 1 < argc) {
      args.x = atoi(argv[++i]);
      x_set = true;
    }
    else if (arg == "-y" and i + 1 < argc) {
      args.y = atoi(argv[++i]);
      y_set = true;
    }
    else if (arg == "-z" and i + 1 < argc) {
      args.z = atoi(argv[++i]);
      z_set = true;
    }
    else if (arg == "-r" and i + 1 < argc) {
      args.radius = atoi(argv[++i]);
      radius_set = true;
    }
    else if (arg == "--codec" and i + 1 < argc) {
      args.codec_type = std::string(argv[++i]) == "fzg" ? FZCodec : Huffman;
    }
    else if (arg == "--help") {
      print_help();
      exit(0);
    }
    else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      print_help();
      exit(1);
    }
  }

  if (not fname_prefix_set or not fname_suffix_set or not from_number_set or not to_number_set or
      not x_set or not y_set or not z_set) {
    std::cerr << "Error: Missing required arguments." << std::endl;
    print_help();
    exit(1);
  }

  return args;
}

std::vector<std::string> construct_file_names(
    const std::string& prefix, const std::string& suffix, int start, int end, int width)
{
  std::vector<std::string> file_names;
  for (int i = start; i < end; ++i) {
    std::ostringstream oss;

    oss << prefix << "." << std::setw(width) << std::setfill('0') << i << "." + suffix;
    file_names.push_back(oss.str());
  }
  return file_names;
}