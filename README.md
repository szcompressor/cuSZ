<img src="https://user-images.githubusercontent.com/10354752/81179956-05860600-8f70-11ea-8b01-856f29b9e8b2.jpg" width="150">

cuSZ: CUDA-Based Error-Bounded Lossy Compressor for Scientific Data
---

cuSZ is a CUDA implementation of the world-widely used [SZ lossy compressor](https://github.com/szcompressor/SZ). It is the first error-bounded lossy compressor on GPUs for scientific data, which significantly improves SZ's throughput in GPU-based heterogeneous HPC systems. 

This document simply introduces how to install and use the cuSZ compressor on NVIDIA GPUs. More details can be found in [doc/cusz-doc.pdf](https://github.com/szcompressor/cuSZ/blob/master/doc/cusz-doc.pdf) [in progress].

Our published paper covers the essential design and implementation, accessible via [ACM DL (open access)](https://dl.acm.org/doi/10.1145/3410463.3414624) or [arXiv](https://arxiv.org/abs/2007.09625).

**Kindly note:** If you mention cuSZ in your paper, please cite using [this entry](https://github.com/szcompressor/cuSZ#citing-cusz).

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

(C) 2020 by Washington State University and Argonne National Laboratory. See [COPYRIGHT](https://github.com/szcompressor/cuSZ/blob/master/LICENSE) in top-level directory.

* Developers: Jiannan Tian, Cody Rivera, Wenyu Gai, Dingwen Tao, Sheng Di, Franck Cappello
* Contributors (alphabetic): Jon Calhoun, Megan Hickman Fulp, Xin Liang, Robert Underwood, Kai Zhao

# set up
## requirements
- `{`Pascal, Volta, Turing, Ampere`}` NVIDIA GPU
- C++14 enabled compiler, GCC 7 onward; CUDA 9.2 onward
  - The table below shows toolchain compatibility; please also refer to [our testbed list](./doc/testbed.md).
  - More reference: 1) [CUDA compiler compatibility](https://gist.github.com/ax3l/9489132), 2) [compilation instruction](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/). 
  - Note that *CUDA version* is referred to as the *toolchain version* (e.g., activated via `module load cuda/<version>`), whereas CUDA *runtime* version can be lower than that.

|      |     |      |      |      |      |      |      |
| ---- | --- | ---- | ---- | ---- | ---- | ---- | ---- |
| gcc  | 7.x | 7.x  | 7.x  | 7.x  | 7.x  | 7.x  | 7.x  |
|      |     | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  |
|      |     |      |      |      | 9.x  | 9.x  | 9.x  |
| CUDA | 9.2 | 10.0 | 10.1 | 10.2 | 11.0 | 11.1 | 11.2 |


## from GitHub

```bash
git clone --recursive git@github.com:szcompressor/cuSZ.git cusz && cd cusz
chmod 755 ./build.py && ./build.py <target> <optional: build type>
```
- For maximal compatibility, use `./build.py compat`. 
- For optimal compilation, use `./build.py <target> <optional: build type>`. 
  - Target names other than `compat` include `p100`, `v100`, `a100`, `pascal`, `turing`, `ampere`.
  - Build types include `release`, `release-profile`, `debug`.
`build.py` automatically builds `cusz`. `sudo make install` can be used to install to system path.

## from Spack

Spack is a multi-platform package manager dedicated for HPC deployment, and it's non-destructive. Currently, deployment of cuSZ requires a workaround on many HPC cluster, see details in [here](./doc/spack-install.md).

# use
## basic use

Type `cusz` or `cusz -h` for instant instructions. Also, a basic use cuSZ is given below.

```bash
./bin/cusz -t f32 -m r2r -e 1.0e-4.0 -i ./data/ex-cesm-CLDHGH -2 3600 1800 -z
           ------ ------ ----------- ------------------------ ------------  |
            dtype  mode  error bound      input datum file    low-to-high  zip

./bin/cusz -i ./data/ex-cesm-CLDHGH.sz -x
           ---------------------------  |
                     sz archive        unzip
```
`-D cesm` specifies preset dataset for demonstration. In this case, it is CESM-ATM, whose dimension is 1800-by-3600, following y-x order. To otherwise specify datum file and input dimensions arbitrarily, we use `-2 3600 1800`, then it becomes

```bash
cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -2 3600 1800 -z
```
The following **essential** arguments are required,

- WORKFLOW: `-z` to compress; `-x` to decompress.
- CONFIG: `-m` to specify error control mode from `abs` (absolute) and `r2r` (relative to value range)
- CONFIG: `-e` to specify error bound
- INPUT: `-i` to specify input file
- INPUT: `-D` to specify demo dataset name or `-{1,2,3}` to dimensions
- OUTPUT: `--opath` to specify output path for both compression and decompression
- LOG: `-V` or `--verbose` to print host and device information

## lossless compression (optional)

gzip (CPU) and/or [nvcomp](https://github.com/NVIDIA/nvcomp) (GPU) lossless compression can be enabled in cuSZ for higher compression ratio by adding ``--gzip`` and/or ``--nvcomp``. For example,

```bash
cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -2 3600 1800 -z --gzip
cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -2 3600 1800 -z --nvcomp
cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -2 3600 1800 -z --gzip --nvcomp
```

We adopt nvcomp's Cascaded compression on GPU. We recommend its project website for more information.

## tuning by overriding
There are also internal a) quant. code representation, b) Huffman codeword representation, and c) chunk size for Huffman coding exposed. Each can be specified with argument options.

- `-Q` or `--quant-byte`  to specify quant. code representation. Options 1, 2 are for 1- and 2-byte, respectively. (Manually specifying this may not result in optimal memory footprint.)
- `-H` or `--huff-byte`  to specify Huffman codeword representation. Options 4, 8 are for 4- and 8-byte, respectively. (Manually specifying this may not result in optimal memory footprint.)
- `-C` or `--huff-chunk`  to specify chunk size for Huffman codec. This should be a sufficiently large number in power-of-2 (`[256|512|1024|...]`) and affects Huffman encoding/decoding performance *significantly*.


## enabling preprocessing
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

1. 2D CESM demo, at 1e-4 relative to value range

	```bash
	# compress
	cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -D cesm -z
	# decompress, use the datum to compress as basename
	cusz -i ./data/ex-cesm-CLDHGH.sz -x
	# decompress, and compare with the original data
	cusz -i ./data/ex-cesm-CLDHGH.sz -x --origin ./data/ex-cesm-CLDHGH
	```
2. 2D CESM demo, specifying output path

	```bash
	mkdir data2 data3
	# output compressed data to `data2`
	cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -D cesm -z --opath data2
	# output decompressed data to `data3`
	cusz -i ./data2/ex-cesm-CLDHGH.sz -x --opath data3
	```
3. 2D CESM demo, with 1-byte (`uint8_t`) and 256 quant. bins

	```bash
	cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -D cesm -z -x -d 256 -Q 1
	```
4. in addition to the previous command, skipping Huffman codec,

	```bash
	cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -D cesm -z -d 256 -Q 1 \
		--skip huffman  # or `-X huffman`
	cusz -i ./data/ex-cesm-CLDHGH.sz -x  # `-d`, `-Q`, `-X` is recorded
	```
5. some application such as EXAFEL preprocessed with binning [^binning] in addition to skipping Huffman codec

	```bash
	** cautious, may not be working as of 0.1.3
	cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -D cesm -z -x \
		-d 256 -Q 1 --pre binning --skip huffman	# or `-p binning`
	```
6. dry-run to get PSNR and to skip real compression or decompression; `-r` also works alternatively to `--dry-run`

	```bash
	# This works equivalently to decompress with `--origin /path/to/origin-datum`
	cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -D cesm --dry-run	# or `-r`
	```

# results
## compression ratio

To calculate compression ratio, please use *size of original data* divided by *size of .sz compressed data*.

## compression throughput

To calculate (de)compression throughput, please follow the below steps to use [`script/sh.parse-nvprof-log`](./script/parse_nvprof_log.sh):
- use `nvprof --log-file <logfile> <cusz cmd>` to generate performance log when (de)compressing
- use `./script/sh.parse-nvprof-log <logfile>` to extract kernel time performance data
- sum up all the numbers between `++++++++` to get the overall (de)compression time in us
- use the original data size divided by the (de)compression time to get the overall (de)compression throughput
- A sample output from on `CLDHGH` (25 MiB, CESM variable) is shown [here](./doc/sample-stat.txt).
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

We provide three small sample data in `data`. To download more SDRBench datasets, please use [`script/sh.download-sdrb-data`](script/sh.download-sdrb-data). 

## sample kernel performance (compression/zip)

Tested on October 8, 2020, on V100; throughput is in the unit of GB/s if not specified otherwise, 

|                    | dual-quant | hist  | codebook | enc. | outlier | OVERALL (w/o c/b) | mem bw (ref) | memcpy (ref) |
| ------------------ | ---------- | ----- | -------- | ---- | ------- | ----------------- | ------------ | ------------ |
| 1D HACC (1.05 GiB) | 312.0      | 400.0 | 0.1 ms   | 57.6 | 278.8   | 37.4              | 900 (HBM2)   | 713.1        |
| 2D CESM (25.7 MiB) | 260.1      | 591.8 | 0.82 us  | 60.1 | 192.0   | 36.5              | 900 (HBM2)   | 713.1        |
| 3D NYX (512 MiB)   | 199.6      | 400.6 | 0.68 us  | 64.1 | 268.4   | 37.3              | 900 (HBM2)   | 713.1        |

A more detailed benchmark can be found at [`doc/benchmark.md`](https://github.com/szcompressor/cuSZ/blob/master/doc/benchmark.md).

## limitations of this version (Jan. 10, 2021)

- cuSZ only supports 4-byte `float` data type. We will support 8-byte `double` data type in the future release. 
- The current Huffman codec consists of optimal (1) histogramming [1], (2) parallel Huffman codebook building [2] of canonical code [3], and suboptimal Huffman encoding. We are woking on a faster high-throughput finer-grained Huffman codec. 
- We are working on host- and device-side API design.
- Please use `-H 8` whenever there is reported error. (The default `-H 4` may not be working for all.) We are working on adapting 4- or 8-byte representation automatically. 
- A performance degradation is expected when handling large-size dataset, e.g., 1-GB or 4-GB 1D HACC. We are working on tuning consistent performance.
- Binning preprocessing is subject to change, currently not available.

# references

[1] 
Gómez-Luna, Juan, José María González-Linares, José Ignacio Benavides, and Nicolás Guil. "An optimized approach to histogram computation on GPU." Machine Vision and Applications 24, no. 5 (2013): 899-908.

[2]
Ostadzadeh, S. Arash, B. Maryam Elahi, Zeinab Zeinalpour, M. Amir Moulavi, and Koen Bertels. "A Two-phase Practical Parallel Algorithm for Construction of Huffman Codes." In PDPTA, pp. 284-291. 2007.

[3]
Klein, Shmuel T. "Space- and Time-Efficient Decoding with Canonical Huffman Trees." In Annual Symposium on Combinatorial Pattern Matching, pp. 65-75. Springer, Berlin, Heidelberg, 1997.

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
This R&D is supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem. This repository is based upon work supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357, and also supported by the National Science Foundation under Grants [CCF-1617488](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1617488), [CCF-1619253](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1619253), [OAC-2003709](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2003709), [OAC-1948447/2034169](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2034169), and [OAC-2003624/2042084](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2042084).

![acknowledgement](https://user-images.githubusercontent.com/5705572/93790911-6abd5980-fbe8-11ea-9c8d-c259260c6295.jpg)
