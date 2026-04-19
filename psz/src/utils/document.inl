/**
 * @file document.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.16.0
 * @date 2020-09-22
 *
 * @copyright (C) 2020 by Washington State University, Argonne National
 * Laboratory See LICENSE in top-level directory
 *
 */

#ifndef ARGUMENT_PARSER_DOCUMENT_HH
#define ARGUMENT_PARSER_DOCUMENT_HH

#include <regex>
#include <string>

const std::string fmt_b("\e[1m");
const std::string fmt_0("\e[0m");

const std::regex bful("@(.*?)@");
const std::string bful_text("\e[1m\e[4m$1\e[0m");
const std::regex bf("\\*(.*?)\\*");
const std::string bf_text("\e[1m$1\e[0m");
const std::regex ul(R"(_((\w|-|\d|\.)+?)_)");
const std::string ul_text("\e[4m$1\e[0m");
const std::regex red(R"(\^\^(.*?)\^\^)");
const std::string red_text("\e[31m$1\e[0m");

std::string  //
Format(const std::string& s)
{
  auto a = std::regex_replace(s, bful, bful_text);
  auto b = std::regex_replace(a, bf, bf_text);
  auto c = std::regex_replace(b, ul, ul_text);
  auto d = std::regex_replace(c, red, red_text);
  return d;
}

static const char psz_short_doc[] =
    "\n"
    "usage: cusz [-zxh] [-i file] [-t dtype] [-m mode] [-e eb] [-l x-y-z] "
    "            (OPTIONAL) [-p predictor] [-a tuning] [-s lossless]"
    "...\n"
    "\n"
    "  z : zip/compress\n"
    "  x : unzip/decompress\n"
    "  h : print full-length help document\n"
    "\n"
    "  i file  : path to input data\n"
    "  t dtype : _f32_ (fp64 to be updated)\n"
    "  m mode  : _abs_, _rel_/_r2r_\n"
    "  e eb    : error bound\n"
    "  l size  : _-l [x[-y[-z]]] (1,2,3-D)\n"
    "  p NAME  : predictor: \"lrz\", \"spl\", \"lrz-zz\", \"lrz-proto\"\n"
    "  a NAME  : auto tuning mode: \"CR-first\", \"RD-first\""
    "  s NAME  : lossless scheme: \"CR\", \"TP\"/\"speed\""
    "\n"
    // "  config list:\n"
    // "    syntax: opt=v, \"kw1=val1,kw1=val2[,...]\"\n"
    // "    + eb     error bound\n"
    // "    + radius The number of quant-codes is 2x radius.\n"
    // "    example: \"--config eb=1e-3,radius=512\"\n"
    // "    example: \"--config eb=1e-3\"\n"
    "  report list: \n"
    "    syntax: opt[=v], \"kw1[=(on|off)],kw2[=(on|off)]\n"
    "    keyworkds: time, quality\n"
    "    example: \"--report time\", \"--report time=off\"\n"
    "\n"
    "example:\n"
    "   cusz -t f32 -m rel -e 1e-4 -i ${CESM} -l 3600-1800 -z --report "
    "time,cr\n"
    "   cusz -i ${CESM}.cusza -x --report time --compare ${CESM}\n"
    "\n"
    "\"cusz -h\" for details.\n";

static const char psz_full_doc[] =
    "*NAME*\n\n"
    "    cuSZ: CUDA-Based Error-Bounded Lossy Compressor for Scientific "
    "Data\n"
    "    The lowercased \"*cusz*\" is the command.\n"
    "\n"
    "*SYNOPSIS*\n\n"
    "    The basic use is listed below,\n"
    "    *cusz* *-t* f32 *-m* rel *-e* 1.0e-4.0 *-i* ${CESM} *-l* "
    "3600-1800 *-z* *--report* time,cr\n"
    "         ^^------ ------ ----------- ---------- ------------  |  ^^\n"
    "         ^^ type   mode  error bound    input    fast-slow   zip "
    "^^\n"
    "\n"
    "    *cusz* *-i* ${CESM} *-x* *--compare* ${CESM} *--report* "
    "time\n"
    "         ^^----------  |   ^^\n"
    "         ^^ archive  unzip ^^\n"
    "\n"
    "    *cusz* *-z* *-i* [filename] *-t* f32 *-m* [eb mode] *-e* [eb] "
    "*-l* [x[-y[-z]]]\n"
    "    *cusz* *-x* *-i* [filename].cusza\n"
    "\n"
    "*OPTIONS*\n\n"
    "  ^^*mandatory::zip*^^\n\n"
    "    *-z* or *--compress* or *--*@z@*ip*\n"
    "\n"
    "    *-m* or *--*@m@*ode* <abs|r2r>  Specify compression mode.\n"
    "        _abs_: absolute mode, eb = input eb\n"
    "        _rel_/_r2r_: relative to value-range, = input eb * value range\n"
    "\n"
    "    *-e* or *--eb* or *--error-bound* [num]\n"
    "        Specify error bound. e.g., _1.23_, _1e-4_, _1.23e-4.56_\n"
    "\n"
    "    *-i* or *--*@i@*nput* [file]\n"
    "\n"
    "    *-d* or *--dict-size* [256|512|1024|...]\n"
    "        Specify a power-of-2 dictionary size/quantization bin number.\n"
    "\n"
    "    *-l* [x[-y[-z]]]  Specify (1|2|3)D (fast-to-slow) data dimension.\n"
    "        A delimiters can be \'x\', \'*\', \'-\', \',\' or \'m\'.\n"
    "\n"
    "  ^^*mandatory::unzip*^^\n\n"
    "    *-x* or *--e*@x@*tract* or *--decompress* or *--unzip*\n"
    "\n"
    "    *-i* or *--*@i@*nput* [compressed file]\n"
    "\n"
    "  ^^*alternative::command*^^\n\n"
    "    *--math-order* == *--zyx* == *--slowest-to-fastest*  Math dimension "
    "order.\n"
    "\n"
    "    *-l* == *--len* == *--xyz* == *--dim3*  CUDA dimension order.\n"
    "\n"
    "  ^^*optional::assessment*^^\n\n"
    "    *--origin* or *--compare* /path/to/origin-data\n"
    "        For verification & get data quality evaluation.\n"
    "        This automatically turns on data-quality assessment report.\n"
    "\n"
    "  ^^*optional::modules*^^\n\n"
    "    *-p* or *--pred* or *--predictor* <lrz|lrz-zz|lrz-proto|spl>\n"
    "        Select from the following predictors: \n"
    "        + _lrz_ or _lorenzo_: (default) Lorenzo predictor.\n"
    "        + _lrz-zz_ or _lorenzo-zigzag_: Lorenzo + ZigZag codec.\n"
    "        + _lrz-proto_ or _lorenzo-proto_: prototype that matches _lrz_.\n"
    "        + _spl_ or _spline_: spline interpolation (3D).\n"
    "\n"
    "    *--hist* or *--histogram* <generic|sparse>\n"
    "        Select from the following histogramming method: \n"
    "        + _generic_: (default) for all quant-code distributions.\n"
    "        + _sparse_: may outperform _generic_ in high-ratio cases.\n"
    "\n"
    "    *-c1* or *--codec* or *--codec1* <hf|fzgcodec>\n"
    "        Select from the following lossless codec: \n"
    "        _hf_ or _huffman_: (default) multibyte Huffman codec.\n"
    "        _fzgcodec_: bitshuffle & de-redundancy in FZ-GPU.\n"
    "\n"
    "  ^^*optional::report::stdout*^^\n\n"
    "    *--report* (option=on/off)-list\n"
    "        syntax: opt[=v], \"kw1[=(on|off)],kw2=[=(on|off)]\n"
    "        keywords: \'time\', \'cr\', \'quality\', \'compressibility\'\n"
    "        example: \"--report time,cr\", \"--report time=off\"\n"
    "\n"
    "  ^^*optional::dump-internal-buffer*^^\n\n"
    "    *--dump* <quant|hist>\n"
    "        Select from the following strings for internal buffers: \n"
    "        _quant_: quant-code binary ^^\"[fname].[mode]_[eb].qt_<u1|u2>\"^^\n"
    "        _hist_ : histogram binary ^^\"[fname].[mode]_[eb].ht[r*2]_u4\"^^\n"
    "\n"
    "  ^^*help::doc*^^\n\n"
    "    *-h* or *--help*  Query documentation.\n"
    "\n"
    "    *-v* or *--version*  Query build number.\n"
    "\n"
    "    *-V* or *--versioninfo* or *--query-env*  Query runtime.\n"
    // "\n"
    // "  ^^*config::string*^^\n\n"
    // "    *-c* or *--config* (option=value)-list\n"
    // "        syntax: opt=v, \"kw1=val1,kw1=val2[,...]\"\n"
    // "        + *eb*=<val>: error bound\n"
    // "        + *cap*=<val>: capacity, number of quant-codes\n"
    "\n"
    "*EXAMPLES*\n\n"
    "  ^^*compression pipelines*^^\n\n"
    "    cuSZ integrates multiple pipelines, specified when compressing. \n\n"
    "    # 1. Lorenzo + Huffman coding (default, balanced)\n"
    "    cusz -t f32 -m rel -e 1e-4 -i ${HURR} -l 500-500-100 -z \\\n"
    "      ^^--predictor lrz --codec hf^^\n"
    "\n"
    "    # 2. Spline-3D + Huffman coding (high-quality)\n"
    "    cusz -t f32 -m rel -e 1e-4 -i ${HURR} -l 500-500-100 -z \\\n"
    "      ^^--predictor spl --codec hf^^\n"
    "\n"
    "    # 3. Lorenzo-variant + FZGPU-coding (fast)\n"
    "    cusz -t f32 -m rel -e 1e-4 -i ${HURR} -l 500-500-100 -z \\\n"
    "      ^^--predictor lrz-zz --codec fzgcodec^^\n"
    "\n"
    "  ^^*testing data*^^\n\n"
    "    Get testing data from Scientific Data Reduction Benchmarks (SDRB)\n"
    "    at https://sdrbench.github.io\n"
    "\n"
    "    # 2D *CESM* example (compression and decompression):\n"
    "    cusz -t f32 -m rel -e 1e-4 -i ${CESM} -l 3600-1800 -z --report "
    "time\n"
    "    cusz -i ${CESM}.cusza -x --report time --compare ${CESM}\n"
    "\n"
    "    # 3D *Hurricane Isabel* example (compression and decompression):\n"
    "    cusz -t f32 -m rel -e 1e-4 -i ${HURR} -l 500-500-100 -z\n"
    "    cusz -i ${HURR}.cusza -x\n";

static const char huff_re_short_doc[] =
    "\n"
    "OVERVIEW: Huffman submodule as standalone program\n"  // TODO from this
                                                           // line on
    "\n"
    "USAGE:\n"
    "  The basic use with demo data is listed below,\n"
    "    ./huff --encode --decode --verify --input ./baryon_density.dat.b16 "
    "\\\n"
    "        -3 512 512 512 --input-rep 16 --huffman-rep 32 --huffman-chunk "
    "2048 --dict-size 1024\n"
    "  or shorter\n"
    "    ./huff -e -d -V -i ./baryon_density.dat.b16 -3 512 512 512 -R 16 -H "
    "32 -C 2048 -c 1024\n"
    "            ^  ^  ^ --------------------------- -------------- ----- "
    "----- ------- -------\n"
    "            |  |  |       input data file         dimension   input "
    "Huff. Huff.   codebook\n"
    "          enc dec verify                                       rep.  "
    "rep.  chunk   size\n"
    "\n"
    "EXAMPLES\n"
    "  Essential:\n"
    "    ./bin/huff -e -d -i ./baryon_density.dat.b16 -3 512 512 512 -R 16 -c "
    "1024\n"
    "    have to input dimension, and higher dimension for a multiplication "
    "of each dim.,\n"
    "    as default values input-rep=16 (bits), huff-rep=32 (bits), "
    "codebook-size=1024 (symbols)\n"
    "\n";

static const char doc_dim_order[] =
    "\n"
    "  Input dimension follows low-to-high (e.g., x-y-z) order.\n"
    "  Taking 2D CESM-ATM as an example, \n"
    "\n"
    "  |<------------------------- x 3600 --------------------------->|    \n"
    "  +--------------------------------------------------------------+  - \n"
    "  |                                                              |  ^ \n"
    "  |                                                              |  | \n"
    "  |              CESM-ATM:    1800x3600 (y-x order)              |  | \n"
    "  |              data name:  <field>_1800_3600                  |  y \n"
    "  |                                                              | 1800 "
    "\n"
    "  |              input:       -l 3600,1800                       |  | \n"
    "  |              input order: -l [x,y]                           |  | \n"
    "  |                                                              |  | \n"
    "  |                                                              |  v \n"
    "  +--------------------------------------------------------------+  - \n"
    "\n"
    "  Taking 3D Hurricane as another example, whose dimensions are\n"
    "  100x500x500, the input is \"-l 500,500,100\".\n";

#endif
