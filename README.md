<img src="https://user-images.githubusercontent.com/10354752/81179956-05860600-8f70-11ea-8b01-856f29b9e8b2.jpg" width="150">

cuSZ: A GPU Accelerated Error-Bounded Lossy Compressor for Scientific Data
---

cuSZ is a CUDA implementation of the world-widely used [SZ lossy compressor](https://github.com/szcompressor/SZ). It is the first error-bounded lossy compressor on GPUs for scientific data, which significantly improves SZ's throughput in GPU-based heterogeneous HPC systems. 

(C) 2020 by Washington State University and Argonne National Laboratory. See COPYRIGHT in top-level directory.

* Major Developers: Jiannan Tian, Cody Rivera, Dingwen Tao, Sheng Di, Franck Cappello
* Other Contributors: Megan Hickman Fulp, Robert Underwood, Kai Zhao, Xin Liang, Jon Calhoun

# citation
**Kindly note**: If you mention cuSZ in your paper, please cite the following reference which covers the whole design and implementation of the latest version of cuSZ.

* Jiannan Tian, Sheng Di, Kai Zhao, Cody Rivera, Megan Hickman Fulp, Robert Underwood, Sian Jin, Xin Liang, Jon Calhoun, Dingwen Tao, Franck Cappello. "[cuSZ: An Efficient GPU-Based Error-Bounded Lossy Compression Framework for Scientific Data](https://arxiv.org/abs/2007.09625)", in Proceedings of the 29th International Conference on Parallel Architectures and Compilation Techniques (PACT), Atlanta, GA, USA, 2020.

This document simply introduces how to install and use the cuSZ compressor on NVIDIA GPUs. More details can be found in doc/cusz-doc.pdf.

# set up
## requirements
- NVIDIA GPU with Pascal, Volta, or Turing microarchitectures 
- CUDA 9.2+ (recommended: CUDA 10.1+) and GCC 7+

## download
```bash
git clone git@github.com:szcompressor/cuSZ.git
```

## compile
```bash
cd cuSZ
export CUSZ_ROOT=$(pwd)
make
sudo make install   # optional given that it's a sudo
# otherwise, without `sudo make install`, `./$(CUSZ_ROOT)/bin/cusz` to execute
```

Commands `cusz` or `cusz -h` are for instant instructions.

## cuSZ as a compressor
### basic use

The basic use cuSZ is given below.

```bash
cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -z -x
       ^  ~~~~~~ ~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~~~~~  ^  ^
       |   mode   error         input datum file        demo   |  |
     dtype        bound                                 data  zip unzip
```
`-D cesm` specifies preset dataset for demonstration. In this case, it is CESM-ATM, whose dimension is 1800-by-3600, following y-x order. To otherwise specify datum file and input dimensions arbitrarily, we use `-2 3600 1800`, then it becomes

```bash
cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -2 3600 1800 -z -x
```
To conduct compression, several input arguments are **necessary**,

- `-z` or `--zip` to compress
- `-x` or `--unzip` to decompress
- `-m` or `--mode` to specify compression mode. Options include `abs` (absolute value) and `r2r` (relative to value range).
- `-e` or `--eb` to specify error bound
- `-i` to specify input datum file
- `-D` to specify demo dataset name or `-{1,2,3}` to input dimensions


### tuning
There are also internal a) quant. code representation, b) Huffman codeword representation, and c) chunk size for Huffman coding exposed. Each can be specified with argument options.

- `-Q` or `--quant-rep`  to specify bincode/quant. code representation. Options `<8|16|32>` are for `uint8_t`, `uint16_t`, `uint32_t`, respectively. (Manually specifying this may not result in optimal memory footprint.)
- `-H` or `--huffman-rep`  to specify Huffman codeword representation. Options `<32|64>` are for `uint32_t`, `uint64_t`, respectively. (Manually specifying this may not result in optimal memory footprint.)
- `-C` or `--huffman-chunk`  to specify chunk size for Huffman codec. Should be a power-of-2 that is sufficiently large (`[256|512|1024|...]`). (This affects Huffman decoding performance *significantly*.)


### extension and use scenarios

#### preprocess 
Some application such as EXAFEL preprocesses with binning [^binning] in addition to skipping Huffman codec.

[^binning]: A current binning setting is to downsample a 2-by-2 cell to 1 point.

#### disabling modules
For EXAFEL, given binning and `uint8_t` have already resulted in a compression ratio of up to 16, Huffman codec may not be needed in a real-world use scenario, so Huffman codec can be skipped with `--skip huffman`.

Decompression can give a full preview of the whole workflow and writing data of the orignal size to the filesystem is long, so writing decompressed data to filesystem can be skipped with `--skip write.x`. 

A combination of modules can be `--skip huffman,write.x`.

Other module skipping for use scenarios are in development.

## cuSZ as an analytical tool

`--dry-run` or `-r` in place of `-a` and/or `-x` enables dry-run mode to get PSNR. This employs the feature of dual-quantization that the decompressed data is guaranteed the same with prequantized data.

## examples

1. run a 2D CESM demo at 1e-4 relative to value range

	```bash
	cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -z -x
	```
2. alternatively, to use full option name,

	```bash
	cusz -f32 --mode r2r --eb 1e-4 --input ./data/sample-cesm-CLDHGH \
		--demo cesm --zip --unzip
	```
3. run a 3D Hurricane Isabel demo at 1e-4 relative to value range

	```bash
	cusz -f32 -m r2r -e 1e-4 -i ./data/sample-hurr-CLOUDf48 -D huricanne -z -x
	```
4. run CESM demo with 1) `uint8_t`, 2) 256 quant. bins,

	```bash
	cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -z -x \
		-d 256 -Q 8
	```
5. in addition to the previous command, if skipping Huffman codec,

	```bash
	cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -z -x \
		-d 256 -Q 8 --skip huffman	# or `-X/-S huffman`
	```
6. some application such as EXAFEL preprocesses with binning [^binning] in addition to skipping Huffman codec

	```bash
	cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -z -x \
		-d 256 -Q 8 --pre binning --skip huffman	# or `-p binning`
	```
7. dry-run to get PSNR and to skip real compression or decompression; `-r` also works alternatively to `--dry-run`

	```bash
	cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm --dry-run	# or `-r`
	```


## tested datasets

We have successfully tested cuSZ on the following datasets from [Scientific Data Reduction Benchmarks](https://sdrbench.github.io/):
- CESM-ATM (Climate simulation)
- Hurricane ISABEL (Weather simulation)
- EXAALT (Molecular dynamics simulation)
- HACC (Cosmology: particle simulation)
- NYX (Cosmology: Adaptive mesh hydrodynamics + N-body cosmological simulation)

## limitations of this version

- For this release, cuSZ can only work for 1) compression, 2) dryrun, 3) decompression right after compression (i.e. put `-z -x` together). We will make decompression standalone in the next release.
- The current integrated Huffman codec runs with efficient histogramming [1], GPU-sequential codebook building, memory-copy style encoding, chunkwise bit deflating, and corresponding canonical Huffman decoding [2], however, the chunkwise bit deflating is not optimal. We are woking on a faster, finer-grained Huffman codec for the next release. 
- We are working on refactoring to support more predictors, preprocessing methods, and compression modes. More functionalities will be released in the next release.
- Please use `-H 64` for HACC dataset because 32-bit representation is not enough for multiple HACC variables. Using `-H 32` will make cuSZ report an error. We are working on automatically adpating 32- or 64-bit representation for different datasets. 
- You may see a performance degradation when handling large-size dataset, such as 1-GB or 4-GB HACC. We are working on autotuning consistent performance.
- Please refer to [_Project Management page_](https://github.com/szcompressor/cuSZ/projects/2) for more todos.  

# `changelog`

September, 2020
- `feature` use cuSPARSE `prune2csr` and `csr2dense` to handle outlier
- `fix` raise error when Huffman code is longer than 32 bits
- `fix` histograming error
- `deploy` fix pSZ
- `feature` integrate parallel build Huffman codebook
- `doc` update help doc
- `doc` update published paper
- `doc` update acknowledgement

August, 2020
- `deploy` `sm_75` for Turing.

July, 2020
- `doc` add a new NSF grant

June, 2020
- `fix` compile with CUDA 9 + gcc 7.3

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
- `deploy` histogramming, 2013 algorithm supersedes naive 2007 algorithm by default
- `feature` add communication of equivalence calculation
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
- `feature` switch to faster histogramming

February, 2020
- `demo` SDRB suite metadata in `SDRB.hh`
- `feature` visualize histogram (`pSZ`)
- `milestone` `PdQ` for compression, Huffman encoding and deflating

# references

[1] 
: Gómez-Luna, Juan, José María González-Linares, José Ignacio Benavides, and Nicolás Guil. "An optimized approach to histogram computation on GPU." Machine Vision and Applications 24, no. 5 (2013): 899-908.

[2]
: Barnett, Mark L. "Canonical Huffman encoded data decompression algorithm." U.S. Patent 6,657,569, issued December 2, 2003.

# acknowledgements
This R&D was supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem. This repository was based upon work supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357, and also supported by the National Science Foundation under Grants [CCF-1617488](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1617488), [CCF-1619253](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1619253), [OAC-2003709](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2003709), [OAC-1948447/2034169](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2034169), and [OAC-2003624/2042084](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2042084).

![acknowledgement](https://user-images.githubusercontent.com/5705572/91359921-58e3c480-e7aa-11ea-98ad-6a27645398cb.jpg)
