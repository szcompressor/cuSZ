![cuSZ logo small](https://user-images.githubusercontent.com/10354752/81179956-05860600-8f70-11ea-8b01-856f29b9e8b2.jpg)

cuSZ: A GPU Accelerated Error-Bounded Lossy Compressor
=

* Major Developers: Jiannan Tian, Dingwen Tao, Sheng Di, Franck Cappello 
* Other Contributors: Cody Rivera, Megan Hickman, Robert Underwood, Kai Zhao, Xin Liang, Jon Calhoun

# set up
## requirements
- NVIDIA GPU with Kepler, Maxwell, Pascal, Volta, or Turing microarchitectures 
- CUDA 9.1+ and GCC 7+ (recommended: CUDA 10.1 + GCC 8)
- CMake 3.11+

## download
```bash
git clone git@github.com:hipdac-lab/cuSZ.git
```

## compile
```bash
cd cuSZ
cmake CMakeLists.txt     # Using cmake to compile cusz for {1,2,3}-D, with Huffman codec
make
```

# run
- `cusz` or `cusz -h` for detailed instruction

### Dual-Quantization with Huffman
- To run a demo
```bash
./cusz
    -f32                # specify data type
    -m <abs|r2r>	# absolute mode OR relative to value range
    -e <num>		# error bound
    -D <demo dataset>
    -z -x		# -z to compress, -x to decompress
    -i <input data>	# input data path
```
and to execute
```bash
./cusz -f32 -m r2r -e 1.23e-4.56 -D cesm -i ./data/sample-cesm-CLDHGH -z -x
```
- We provide a dry-run mode to quickly get the summary of compression quality (without Huffman coding and decompression)
```bash
./cusz -f32 -m r2r -e 1.23e-4.56 -D cesm -i ./data/sample-cesm-CLDHGH -r
```
- To run cuSZ on any given data field with arbitrary input size(s)
```bash
./cusz
    -f32                # specify datatype
    -m <abs|r2r>	# absolute mode OR relative to value range
    -e <num>		# error bound
    -z -x		# -z to compress, -x to decompress
    -i <input data>	# input data path
    -2 nx ny		# dimension(s) with low-to-high order
```
and to execute
```bash
./cusz -f32 -m r2r -e 1.23e-4.56 -i ./data/sample-cesm-CLDHGH -2 3600 1800 -z -x
```

- Specify `--skip huffman` to skip Huffman encoding and decoding.
- Specify `--verify huffman` to verify the correctness of Huffman codec in decompression (refer to `TODO`: autoselect Huffman codeword rep).
- Specify `-p binning` or `--pre binning` to do binning *prior to* PdQ/compression or dry-run.
- Specify `-Q <8|16|32>` for quantization code representation.
- Specify `-H <32|64>` for Huffman codeword representation.
- Specify `-C <power-of-2>` for Huffman chunk size. 
    - Note that the chunk size significantly affects the throughput, and we estimate that it should match/be closed to some maximum hardware supported number of concurrent threads for optimal performance.
- The integrated Huffman codec runs with efficient histogramming [1], GPU-sequantial codebook building, memory-copy style encoding, chunkwise bit concatenation, and corresponding canonical Huffamn decoding [2].

# TODO List
- `f64` support
- single archive
- autoselect Huffman codeword representation

# `changelog`
May, 2020
- `feature` add `--skip huffman` and `--verify huffman` options
- `feature` add binning as preprocessing
- `prototype` use `cuSparse` to transform `outlier` to dense format
- `feature` add `argparse` to check and parse argument inputs
- `refactor` add CUDA wrappers (e.g., `mem::CreateCUDASpace`)

April, 2020
- `feature` add concise and detailed help doc
- `deploy` `sm_61` (e.g., P1000) and `sm_70` (e.g., V100) binary
- `feature` add dry-run mode
- `refactor` merge cuSZ and Huffman codec in driver program
- `perf` 1D PdQ (and reverse PdQ) `blockDim` set to 32, throughput changed from 2.7 GBps to 16.8 GBps
- `deploy` histograming, 2013 algorithm supersedes naive 2007 algorithm by default
- `feature` add communication of equivalance calculation
- `feature` use cooperative groups (CUDA 9 required) for canonical Huffman codebook
- `perf` faster initializing shared memory for PdQ, from 150 GBps to 200 GBps
- `feature` add Huffman inflating/decoding
- `refactor` merge 1,2,3-D cuSZ
- `feature` set 32- and 64-bit as internal Huffman codeword representation
- `feature` now use arbitrary multiple-of-8-bit for quantization code
- `feature` switch to canonical Huffman code for decoding

March, 2020
- `perf` tuning thread number for Huffman deflating and inflating
- `feature` change freely to 32bit intermediate Huffman code representation
- `demo` add EXAFEL demo
- `feature` switch to faster histograming

February, 2020
- `demo` SDRB suite metadata in `SDRB.hh`
- `feature` visualize histogram (`pSZ`)
- `milestone` `PdQ` for compression, Huffman encoding and deflating

# reference
 - [1] Gómez-Luna, Juan, José María González-Linares, José Ignacio Benavides, and Nicolás Guil. "An optimized approach to histogram computation on GPU." Machine Vision and Applications 24, no. 5 (2013): 899-908.
 - [2] Barnett, Mark L. "Canonical Huffman encoded data decompression algorithm." U.S. Patent 6,657,569, issued December 2, 2003.
 
# acknowledgement
This R&D was supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem. This repository was based upon work supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357, and also supported by the National Science Foundation under Grant No. 1948447.
