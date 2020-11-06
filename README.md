<img src="https://user-images.githubusercontent.com/10354752/81179956-05860600-8f70-11ea-8b01-856f29b9e8b2.jpg" width="150">

cuSZ: CUDA-Based Error-Bounded Lossy Compressor for Scientific Data
---

cuSZ is a CUDA implementation of the world-widely used [SZ lossy compressor](https://github.com/szcompressor/SZ). It is the first error-bounded lossy compressor on GPUs for scientific data, which significantly improves SZ's throughput in GPU-based heterogeneous HPC systems. 

This document simply introduces how to install and use the cuSZ compressor on NVIDIA GPUs. More details can be found in [doc/cusz-doc.pdf](https://github.com/szcompressor/cuSZ/blob/master/doc/cusz-doc.pdf) [in progress].

Our published paper covers the essential design and implementation, accessible via [ACM DL (open access)](https://dl.acm.org/doi/10.1145/3410463.3414624) or [arXiv](https://arxiv.org/abs/2007.09625).

**Kindly note:** If you mention cuSZ in your paper, please cite using [this entry](https://github.com/szcompressor/cuSZ#citing-cusz).

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

(C) 2020 by Washington State University and Argonne National Laboratory. See [COPYRIGHT](https://github.com/szcompressor/cuSZ/blob/master/LICENSE) in top-level directory.

* Developers: Jiannan Tian, Cody Rivera, Dingwen Tao, Sheng Di, Franck Cappello
* Contributors (alphabetic): Jon Calhoun, Megan Hickman Fulp, Wenyu Gai, Xin Liang, Robert Underwood, Kai Zhao

# set up
## requirements
- NVIDIA GPU with Pascal, Volta, or Turing microarchitectures 
- Minimum: CUDA 9.2+ and GCC 7.3+ (with C++14 support)
  - The table below shows our tested GPUs, CUDA versions, and compilers.
  - Note that CUDA version here refers to the toolchain verion (e.g., activiated CUDA via `module load`), whereas CUDA runtime version (according to SM) can be lower than that.
  - Please refer to [link](https://gist.github.com/ax3l/9489132) for more details about different CUDA versions and their required compilers.
  
| GPU       | microarch | SM  | CUDA version | gcc version |
| --------- | --------- | --- | ------------ | ----------- |
| P100      | Pascal    | 60  | 10.1         | 7.3         |
| P2000M    | Pascal    | 61  | 11.0         | 7.5         |
| V100      | Volta     | 70  | 10.2         | 7.3/8.4     |
|           |           |     | 9.2          | 7.3         |
| RTX 8000  | Turing    | 75  | 10.1         | 7.4         |
| RTX 5000  | Turing    | 75  | 10.1         | 7.3/8.3     |
| RTX 2060S | Turing    | 75  | 11.0/11.1    | 9.3         |


## from GitHub
```bash
git clone git@github.com:szcompressor/cuSZ.git cusz
cd cusz && export CUSZ_ROOT=$(pwd)
make   # can use ${CUSZ_ROOT}/bin/cusz to execute
sudo make install  # requires elevated permission
```
## from Spack

Spack is a multi-platform package manager dedicated for HPC deployment, and it's non-destructive. Currently, deployment of cuSZ requies a workaround on many HPC cluster, see details in [here](./doc/spack-install.md).

# use
## basic use

Type `cusz` or `cusz -h` for instant instructions. Also, a basic use cuSZ is given below.

```bash
./bin/cusz -f32 -m r2r -e 1.0e-4.0 -i ./data/sample-cesm-CLDHGH -D cesm -z
             |  ------ ----------- ---------------------------- -------  |
           dtype mode  error bound           input datum          demo   zip

./bin/cusz -i ./data/sample-cesm-CLDHGH.sz -x
           -------------------------------  |
                     sz archive            unzip
```
`-D cesm` specifies preset dataset for demonstration. In this case, it is CESM-ATM, whose dimension is 1800-by-3600, following y-x order. To otherwise specify datum file and input dimensions arbitrarily, we use `-2 3600 1800`, then it becomes

```bash
cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -2 3600 1800 -z
```
To conduct compression, several input arguments are **necessary**,

- `-z` or `--zip` to compress
- `-x` or `--unzip` to decompress
- `-m` or `--mode` to specify compression mode. Options include `abs` (absolute value) and `r2r` (relative to value range)
- `-e` or `--eb` to specify error bound
- `-i` to specify input datum file
- `-D` to specify demo dataset name or `-{1,2,3}` to input dimensions
- `--opath` to specify output path for both compression and decompresson
- `-V` or `--verbose` to print host and device information

## tuning
There are also internal a) quant. code representation, b) Huffman codeword representation, and c) chunk size for Huffman coding exposed. Each can be specified with argument options.

- `-Q` or `--quant-rep`  to specify bincode/quant. code representation. Options `<8|16|32>` are for `uint8_t`, `uint16_t`, `uint32_t`, respectively. (Manually specifying this may not result in optimal memory footprint.)
- `-H` or `--huffman-rep`  to specify Huffman codeword representation. Options `<32|64>` are for `uint32_t`, `uint64_t`, respectively. (Manually specifying this may not result in optimal memory footprint.)
- `-C` or `--huffman-chunk`  to specify chunk size for Huffman codec. This should be a sufficiently large number in power-of-2 (`[256|512|1024|...]`) and affects Huffman encoding/decoding performance *significantly*.


## with preprocessing
Some application such as EXAFEL preprocesses with binning [^binning] in addition to skipping Huffman codec.

[^binning]: A current binning setting is to downsample a 2-by-2 cell to 1 point.

## disabling modules
For EXAFEL, given binning and `uint8_t` have already resulted in a compression ratio of up to 16, Huffman codec may not be needed in a real-world use scenario, so Huffman codec can be skipped with `--skip huffman`.

Decompression can give a full preview of the whole workflow and writing data of the orignal size to the filesystem is long, so writing decompressed data to filesystem can be skipped with `--skip write.x`. 

A combination of modules can be `--skip huffman,write.x`.

Other module skipping for use scenarios are in development.

## use as an analytical tool

`--dry-run` or `-r` in place of `-z` and/or `-x` enables dry-run mode to get PSNR. This employs the feature of dual-quantization that the decompressed data is guaranteed the same with prequantized data.

# hands-on examples

1. run a 2D CESM demo at 1e-4 relative to value range

	```bash
	# compress
	cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -z
	# decompress, use the datum to compress as basename
	cusz -i ./data/sample-cesm-CLDHGH.sz -x
	# decompress, and compare with the original data
	cusz -i ./data/sample-cesm-CLDHGH.sz -x --origin ./data/sample-cesm-CLDHGH
	```
2. runa 2D CESM demo with specified output path

	```bash
	mkdir data2 data3
	# output compressed data to `data2`
	cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -z --opath data2
	# output decompressed data to `data3`
	cusz -i ./data2/sample-cesm-CLDHGH.sz -x --opath data3
	```
3. run CESM demo with `uint8_t` and 256 quant. bins

	```bash
	cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -z -x -d 256 -Q 8
	```
4. in addition to the previous command, if skipping Huffman codec,

	```bash
	cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -z -d 256 -Q 8 \
		--skip huffman  # or `-X huffman`
	cusz -i ./data/sample-cesm-CLDHGH.sz -x  # `-d`, `-Q`, `-X` is recorded
	```
5. some application such as EXAFEL preprocesses with binning [^binning] in addition to skipping Huffman codec

	```bash
	** cautious, may not be working as of 0.1.3
	cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -z -x \
		-d 256 -Q 8 --pre binning --skip huffman	# or `-p binning`
	```
6. dry-run to get PSNR and to skip real compression or decompression; `-r` also works alternatively to `--dry-run`

	```bash
	# This works equivalently to decompress with `--origin /path/to/origin-datum`
	cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm --dry-run	# or `-r`
	```

# results
## compression ratio

To calculate compression ratio, please use *size of original data* divided by *size of .sz compressed data*.

## compression throughput

To calculate (de)compression throughput, please follow the below steps to use our bash script [`cuSZ/script/parse_nvprof_log.sh`](./script/parse_nvprof_log.sh):
- use `nvprof --log-file [name_of_logfile.txt]` before `cusz` to dump the performance data when (de)compressing
- use `bash parse_nvprof_log.sh [name_of_logfile.txt]` to filter out the unnecessary performance data
- sum up all the numbers between `++++++++++` to get the overall (de)compression time in us
- use the original data size divided by the (de)compression time to get the overall (de)compression throughput
- A sample outpput from `bash parse_nvprof_log.sh <log file>` on the CESM variable `CLDHGH` (25 MiB) is shown [here](./doc/sample-stat.txt).
- From the [sample](./doc/sample-stat.txt), the compression and decompression times are 733.47 us (w/o c/b) and 1208.19 us, respectively. So, the compression and decompression throughputs are 31.4 GB/s and 20.7 GB/s, respectively.

**Please note that cuSZ's performance might be dropped for a single large input file (e.g., in several Gigabytes) because of current coarse-grained deflating in Huffman codec mentioned in [limitations of this version](https://github.com/szcompressor/cuSZ#limitations-of-this-version-011).**

# tests by team
## tested datasets

We have successfully tested cuSZ on the following datasets from [Scientific Data Reduction Benchmarks](https://sdrbench.github.io/) (SDRBench):
| dataset                                                                 | dim. | description                                                  |
| ----------------------------------------------------------------------- | ---- | ------------------------------------------------------------ |
| [EXAALT](https://gitlab.com/exaalt/exaalt/-/wikis/home)                 | 1D   | molecular dynamics simulation                                |
| [HACC](https://www.alcf.anl.gov/files/theta_2017_workshop_heitmann.pdf) | 1D   | cosmology: particle simulation                               |
| [CESM-ATM](https://www.cesm.ucar.edu)                                   | 2D   | climate simulation                                           |
| [EXAFEL](https://lcls.slac.stanford.edu/exafel)                         | 2D   | images from the LCLS instrument                              |
| [Hurricane ISABEL](http://vis.computer.org/vis2004contest/data.html)    | 3D   | weather simulation                                           |
| [NYX](https://amrex-astro.github.io/Nyx/)                               | 3D   | adaptive mesh hydrodynamics + N-body cosmological simulation |

We provide three small sample data in `cuSZ/data` directory. To download more SDRBench datasets, please use our bash script `cuSZ/script/download-sdrb-data.sh`. 

## sample kernel performance (compression/zip)

As of October 8, 2020, 
|                    |          |                     | dual-quant | hist  | codebook | enc. | outlier | OVERALL (w/o c/b) | mem bw (ref) | memcpy (ref) |
| ------------------ | -------- | ------------------- | ---------- | ----- | -------- | ---- | ------- | ----------------- | ------------ | ------------ |
| 1D HACC (1.05 GiB) | **V100** | *throughput* (GB/s) | 312.0      | 400.0 | 0.1 ms   | 57.6 | 278.8   | 37.4              | 900 (HBM2)   | 713.1        |
| 2D CESM (25.7 MiB) | **V100** | *throughput* (GB/s) | 260.1      | 591.8 | 0.82 us  | 60.1 | 192.0   | 36.5              | 900 (HBM2)   | 713.1        |
| 3D NYX (512 MiB)   | **V100** | *throughput* (GB/s) | 199.6      | 400.6 | 0.68 us  | 64.1 | 268.4   | 37.3              | 900 (HBM2)   | 713.1        |
A more detailed benchmark can be found at [`doc/benchmark.md`](https://github.com/szcompressor/cuSZ/blob/master/doc/benchmark.md).

**Please note that if the performance you get is much lower than what we show above, please use `-C` option to change the chunk size for Huffman codec. For  example, we use `-C 16384` for 1D HACC data in the above test.**

## limitations of this version (0.1.3)

- For this release, cuSZ only supports 32-bit `float`-type datasets. We will support 64-bit `double`-type datasets in the future release. 
- The current integrated Huffman codec runs with efficient histogramming [1], parallel Huffman codebook building [2], memory-copy style encoding, chunkwise bit deflating, and efficient Huffman decoding using canonical codes [3]. However, the chunkwise bit deflating is not optimal, so we are woking on a faster, finer-grained Huffman codec for the future release. 
- We are working on refactoring to support more predictors, preprocessing methods, and compression modes. More functionalities will be released in the next release.
- Please use `-H 64` for HACC dataset because 32-bit representation is not enough for multiple HACC variables. Using `-H 32` will make cuSZ report an error. We are working on automatically adpating 32- or 64-bit representation for different datasets. 
- You may see a performance degradation when handling large-size dataset, such as 1-GB or 4-GB HACC. We are working on autotuning consistent performance.
- Please refer to [_Project Management page_](https://github.com/szcompressor/cuSZ/projects/2) for more TODO details.  

# references

[1] 
Gómez-Luna, Juan, José María González-Linares, José Ignacio Benavides, and Nicolás Guil. "An optimized approach to histogram computation on GPU." Machine Vision and Applications 24, no. 5 (2013): 899-908.

[2]
Ostadzadeh, S. Arash, B. Maryam Elahi, Zeinab Zeinalpour, M. Amir Moulavi, and Koen Bertels. "A Two-phase Practical Parallel Algorithm for Construction of Huffman Codes." In PDPTA, pp. 284-291. 2007.

[3]
Klein, Shmuel T. "Space-and time-efficient decoding with canonical huffman trees." In Annual Symposium on Combinatorial Pattern Matching, pp. 65-75. Springer, Berlin, Heidelberg, 1997.

# citing cuSZ
```bibtex
@inproceedings{cusz2020,
     author = {Tian, Jiannan and Di, Sheng and Zhao, Kai and Rivera, Cody and Fulp, Megan Hickman and Underwood, Robert and Jin, Sian and Liang, Xin and Calhoun, Jon and Tao, Dingwen and Cappello, Franck},
      title = {cuSZ: An Efficient GPU-Based Error-Bounded Lossy Compression Framework for Scientific Data},
       year = {2020},
       isbn = {9781450380751},
  publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3410463.3414624},
        doi = {10.1145/3410463.3414624},
  booktitle = {Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques},
      pages = {3–15},
   numpages = {13},
   keywords = {cuda, gpu, scientific data, lossy compression, performance},
   location = {Virtual Event, GA, USA},
     series = {PACT '20}
}
```

# acknowledgements
This R&D was supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem. This repository was based upon work supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357, and also supported by the National Science Foundation under Grants [CCF-1617488](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1617488), [CCF-1619253](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1619253), [OAC-2003709](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2003709), [OAC-1948447/2034169](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2034169), and [OAC-2003624/2042084](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2042084).

![acknowledgement](https://user-images.githubusercontent.com/5705572/93790911-6abd5980-fbe8-11ea-9c8d-c259260c6295.jpg)
