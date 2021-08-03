<img src="https://user-images.githubusercontent.com/10354752/81179956-05860600-8f70-11ea-8b01-856f29b9e8b2.jpg" width="150">

cuSZ: CUDA-Based Error-Bounded Lossy Compressor for Scientific Data
---

cuSZ is a CUDA implementation of the world-widely used [SZ lossy compressor](https://github.com/szcompressor/SZ). It is the *first* error-bounded lossy compressor on GPUs for scientific data, which significantly improves SZ's throughput in GPU-based heterogeneous HPC systems. 
This document introduces the installation and use of cuSZ on NVIDIA GPUs. 

Our published papers cover the essential design and implementation, accessible via 
- **PACT '20 cuSZ**, via 1) [local copy](doc/PACT'20-cusz.pdf), 2) [ACM entry (open access)](https://dl.acm.org/doi/10.1145/3410463.3414624), or 3) [arXiv-2007.09625](https://arxiv.org/abs/2007.09625).
- **CLUSTER '21 cuSZ+**, via 1) [local copy](doc/CLUSTER'21-cusz+.pdf), or 2) [arXiv-2105.12912](https://arxiv.org/abs/2105.12912) 
  - `doi` will be ready along with the inproceedings.
  - We are working on integrating the sparsity-aware compression.

*Kindly note:* If you mention cuSZ in your paper, please cite using [these entries](https://github.com/szcompressor/cuSZ#citing-cusz).

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

(C) 2020 by Washington State University and Argonne National Laboratory. See [COPYRIGHT](https://github.com/szcompressor/cuSZ/blob/master/LICENSE) in top-level directory.

* developers: Jiannan Tian, Cody Rivera, Wenyu Gai, Dingwen Tao, Sheng Di, Franck Cappello
* contributors (alphabetic): Jon Calhoun, Megan Hickman Fulp, Xin Liang, Robert Underwood, Kai Zhao
* Special thanks to Dominique LaSalle (NVIDIA) for serving as Mentor in Argonne GPU Hackaton 2021!

- [set up](#set-up)
	- [requirements](#requirements)
	- [compile from source code](#compile-from-source-code)
- [use](#use)
	- [basic use](#basic-use)
	- [advanced use](#advanced-use)
- [hands-on examples](#hands-on-examples)
- [FAQ](#faq)
	- [How does SZ/cuSZ work?](#how-does-szcusz-work)
	- [Why is there multibyte symbol for Huffman?](#why-is-there-multibyte-symbol-for-huffman)
	- [What differs CPU-SZ and cuSZ in data quality and compression ratio?](#what-differs-cpu-sz-and-cusz-in-data-quality-and-compression-ratio)
	- [How to know compression ratio?](#how-to-know-compression-ratio)
	- [How to know the performance?](#how-to-know-the-performance)
- [tested by our team](#tested-by-our-team)
	- [datasets](#datasets)
	- [limitations (build 2021-07-13.2)](#limitations-build-2021-07-132)
- [citing cuSZ](#citing-cusz)
	- [PACT '20, cuSZ](#pact-20-cusz)
	- [CLUSTER '21, cuSZ+](#cluster-21-cusz)
- [acknowledgements](#acknowledgements)

# set up
## requirements
- `{`Pascal, Volta, Turing, Ampere`}` NVIDIA GPU
- cmake 3.18 onward
- C++14 enabled compiler, GCC 7 onward; CUDA 9.2 onward (11.x is recommended)
  - The table below shows toolchain compatibility; please also refer to [our testbed list](./doc/testbed.md).
  - more reference: 1) [CUDA compilers](https://gist.github.com/ax3l/9489132), 2) [CUDA archs](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/). 

|      |     |      |      |      |      |      |      |      |      |
| ---- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| gcc  | 7.x | 7.x  | 7.x  | 7.x  | 7.x  | 7.x  | 7.x  | 7.x  |      |
|      |     | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  |      |
|      |     |      |      |      | 9.x  | 9.x  | 9.x  | 9.x  | 9.x  |
| CUDA | 9.2 | 10.0 | 10.1 | 10.2 | 11.0 | 11.1 | 11.2 | 11.3 | 11.4 |


## compile from source code

```bash
git clone https://github.com/szcompressor/cuSZ.git cusz && cd cusz
chmod 755 ./build.py && ./build.py <target> <optional: build type>
```
- For the maximum compatibility, use `./build.py compat`. 
- For optimal compilation, use `./build.py <target> <optional: build type>`. 
  - Target names other than `compat` include `p100`, `v100`, `a100` and general `pascal`, `turing`, `ampere`.
  - Build types include `release` (default), `release-profile` (enabling `-lineinfo`) and `debug` (enabling `-G`).
- `build.py` automatically builds and installs `cusz` binary to `<workspace>/bin`.


# use
## basic use

Type `cusz` or `cusz -h` for instant instructions. We give a basic use below.

```bash
## cusz -t <dtype> -m <mode> -e <error bound> -i <input> -l <size> -z (compression) --report time[,quality[,...]]
./bin/cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -l 3600,1800 -z --report time
## cusz -i <cusz archive> -x [--compare -i <origin>] (decompresion)
./bin/cusz -i ./data/ex-cesm-CLDHGH.sz -x
./bin/cusz -i ./data/ex-cesm-CLDHGH.sz -x --compare ./data/ex-cesm-CLDHGH.sz --report time,quality
```
We use 1800-by-3600 (y-x order) CESM-ATM CLDHGH for demonstration, which is in preset and `-D cesm` can be alternatively used . Type `cusz` or `cusz -h` to look up the presets.

```bash
cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH --demo cesm -z
```
The following **essential** arguments are required,

- `-z` to compress; `-x` to decompress.
- `-m` to specify error control mode from `abs` (absolute) and `r2r` (relative to value range)
- `-e` to specify error bound
- `-i` to specify input file
- `-l <size>` to specify dimensions; alternatively `--demo <dataset>` to load predefined size

## advanced use

**disabling modules**

- (in progress) To export quant-code, use `--skip huffman`
- For non-IO use, we can skip writing to disk during decompression using `--skip write2disk`. 
- A combination of modules can be `--skip huffman,write2disk`.

**additional lossless compression**

```bash
cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH -l 3600,1800 -z --gzip
```
**changing internal parameter**

syntax: `-c` or `--config quantbyte=(1|2),huffbyte=(4|8)`

- `quantbyte` to specify quant. code representation. Options `{1,2}` are for 1- and 2-byte, respectively. 
- `huffbyte` to specify Huffman codeword representation. Options `{4,8}` are for 4- and 8-byte, respectively. (Manually specifying this may not result in optimal memory footprint.)


**`dry-run` mode**

`--dry-run` or `-r` in place of `-z` and/or `-x` enables dry-run mode to get PSNR. This employs the feature of dual-quantization that the decompressed data is guaranteed the same as the prequantized data.

# hands-on examples

1. 2D CESM demo, at 1e-4 relative to value range

	```bash
	# compress
	cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH --demo cesm -z \
		--report time
	# decompress
	cusz -i ./data/ex-cesm-CLDHGH.sz -x --report time
	# decompress and compare with the original data
	cusz -i ./data/ex-cesm-CLDHGH.sz -x --compare ./data/ex-cesm-CLDHGH \
		--report time,quality
	```
2. 2D CESM demo, specifying output path

	```bash
	mkdir data2 data3
	# output compressed data to `data2`
	cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH --demo cesm -z --opath data2
	# output decompressed data to `data3`
	cusz -i ./data2/ex-cesm-CLDHGH.sz -x --opath data3
	```
3. 2D CESM demo, with 1-byte (`uint8_t`) and 256 quant. bins

	```bash
	cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH --demo cesm -z \
		--config cap=256,quantbyte=1 \
		--report time
	```
4. in addition to the previous command, skipping Huffman codec,

	```bash
	cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH --demo cesm -z \
		--config cap=256,quantbyte=1 \
		--skip huffman
	cusz -i ./data/ex-cesm-CLDHGH.sz -x
	```
5. dry-run to get PSNR and to skip real compression or decompression; `-r` or `--dry-run`

	```bash
	# This works equivalently to decompress with `--origin /path/to/origin-datum`
	cusz -t f32 -m r2r -e 1e-4 -i ./data/ex-cesm-CLDHGH --demo cesm -r
	```

# FAQ

`*` There are implementation differences between CPU-SZ and cuSZ. Some technical detail is simplified. Please refer to our academic papers for more information.  
## How does SZ/cuSZ work?

Prediction-based SZ algorithm comprises of 4 major parts

0. User specifies error-mode (e.g., absolute value (`abs`), or relative to data value magnitude (`r2r`) and error-bound.
1. Prediction errors are quantized/integerized in units of input error-bound (*quant-code*). A selected range of quant-codes are stored, whereas the out-of-range codes are otherwise gathered as *outlier*.
3. The in-range quant-codes are fed into Huffman encoder. A Huffman symbol may be represented in multiple bytes.
4. (CPU-only) additional DEFLATE method is applied to exploit repeated patterns. As of CLUSTER '21 cuSZ+ work, an RLE method performs a similar pattern-exploiting.

## Why is there multibyte symbol for Huffman?

The principle of Huffman coding is to guarantee high-frequency symbols with fewer bits. To be specific, given arbitray pairs of (symbol, frequency)-s, (*s<sub>i</sub>*, *f<sub>i</sub>*) and 
(*s<sub>j</sub>*, *f<sub>j</sub>*), the assigned codeword *c<sub>i</sub>* and *c<sub>j</sub>*, respectively, are guaranteed to have len(*c<sub>i</sub>*) is no greater than len(*c<sub>j</sub>*) if *f<sub>i</sub>* is no less than *f<sub>j</sub>*.

The combination of *n* single-byte does not reflect the fact that that quant-code that represents the `+/-1` error-bound should be of the highest frequency. For example, an enumeration with 1024 symbols can cover 99.8% error-control code (the rest 0.2% can result in much more bits in codewords), among which the most frequent symbol can dominate at over 90%. If singlebyte symbols are used, `0x00` from bytes near MSB makes

1. the highest frequency not properly represented, and
2. the pattern-exploiting harder. For example, `0x00ff,0x00ff` is otherwise interpreted as `0x00,0xff,0x00,0xff`.  


## What differs CPU-SZ and cuSZ in data quality and compression ratio?

CPU-SZ offers a rich set of compression features and is far more mature than cuSZ. (1) CPU-SZ has preprocessing, more compression mode (e.g., point-wise) and autotuning. (2) CPU-SZ has Lorenzo predictor and Linear Regression, whereas cuSZ has Lorenzo (we are working on new predictors).

1. They share the same Lorenzo predictor. However, many factors affect data quality (and can be quality optimizers).
   1. preprocessing such as log transform and point-wise transform
   2. PSNR as a goal to autotune eb
   3.  initial values from which we predict border values (as if padding). cuSZ predicts from zeros while SZ determines optimal values for, e.g., application-specific metrics. Also note that cuSZ compression can result in a significantly higher PSNR than SZ (with the same eb, see Table 8 on page 10 of PACT '20 paper), but it is not necessarily better when it comes to applications.
   5.  The PSNR serves as a generic metric: SZ guarantees a lower bound of PSNR when the eb is relative to the data range, e.g., 64 for 1e-3, 84 for 1e-4.
2. The linear scaling can be the same. SZ has an extra optimizer to decide the linear scaling range $[-r, +r]$; out-of-range quantization values are outliers. This is to optimize the compression ratio.
3. Currently, the Huffman encoding is the same except cuSZ partitions data (therefore, it has overhead in padding bits and partitioning metadata).


|        | preprocess | Lorenzo predictor | other predictors | Huffman | DEFLATE     |
| ------ | ---------- | ----------------- | ---------------- | ------- | ----------- |
| CPU-SZ | x          | x                 | x                | x       | x           |
| cuSZ   | TBD        | x, dual-quant     | TBD              | x       | alternative |
## How to know compression ratio?

Expected to be known at compression time. TBD. 

## How to know the performance?


1. prior to CUDA 11: `nvprof <cusz command>`
2. CUDA 11 onward: `nsys profile --stat true <cusz command>`
3. enable `--report time` in CLI
4. A sample benchmark is shown at [`doc/benchmark.md`](https://github.com/szcompressor/cuSZ/blob/master/doc/benchmark.md). To be updated.

# tested by our team
## datasets

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

## limitations (build 2021-07-13.2)

- cuSZ supports 4-byte `float` data type. We will support 8-byte `double` data type in the future release. 
- To integrate faster Huffman codec.
- To release API.
- The default 4-byte Huffman symbol representation may break; `--config huffbyte=8` is needed.
- A performance degradation is expected when handling large-size datasets, e.g., 1-GB or 4-GB 1D HACC. We are working on tuning consistent performance.
- To update with preprocessing such as binning and log-transform.


# citing cuSZ

## PACT '20, cuSZ
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

## CLUSTER '21, cuSZ+
`doi` to be updated; a *temporary* entry is listed below.
```bibtex
@misc{cuszplus2021,
      title = {cuSZ+: Optimizing Error-Bounded Lossy Compression for Scientific Data on GPUs}, 
     author = {Jiannan Tian and Sheng Di and Xiaodong Yu and Cody Rivera and Kai Zhao and Sian Jin and Yunhe Feng and Xin Liang and Dingwen Tao and Franck Cappello},
      year = {2021},
    eprint = {2105.12912},
	  note = {accepted by IEEE CLUSTER '21}
archivePrefix = {arXiv},
 primaryClass = {cs.DC}
}
```

# acknowledgements
This R&D is supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem. This repository is based upon work supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357, and also supported by the National Science Foundation under Grants [CCF-1617488](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1617488), [CCF-1619253](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1619253), [OAC-2003709](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2003709), [OAC-1948447/2034169](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2034169), and [OAC-2003624/2042084](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2042084).

![acknowledgement](https://user-images.githubusercontent.com/5705572/93790911-6abd5980-fbe8-11ea-9c8d-c259260c6295.jpg)
