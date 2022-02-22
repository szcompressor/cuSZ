<h3 align="center"><img src="https://user-images.githubusercontent.com/10354752/81179956-05860600-8f70-11ea-8b01-856f29b9e8b2.jpg" width="150"></h3>

<h3 align="center">
A CUDA-Based Error-Bounded Lossy Compressor for Scientific Data
</h3>

<p align="center">
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
</p>

cuSZ is a CUDA implementation of the world-widely used [SZ lossy compressor](https://github.com/szcompressor/SZ). It is the *first* error-bounded lossy compressor on GPU for scientific data, and it aims to improve SZ's throughput significantly on GPU-based heterogeneous HPC systems. 
<!-- This document introduces the installation and use of cuSZ on NVIDIA GPUs.  -->

Our published papers cover the essential design and implementation.
- **PACT '20: cuSZ**, [via local copy](doc/PACT'20-cusz.pdf), [via ACM](https://dl.acm.org/doi/10.1145/3410463.3414624), [via arXiv](https://arxiv.org/abs/2007.09625)
  - framework: (fine-grained) *N*-D prediction-based error-controling "construction" + (coarse-grained) lossless encoding
- **CLUSTER '21: cuSZ+**, [via local](doc/CLUSTER'21-cusz+.pdf), [via IEEEXplore](https://doi.ieeecomputersociety.org/10.1109/Cluster48925.2021.00047})
  - optimization in throughput, featuring fine-grained *N*-D "reconstruction"
  - optimization in compression ratio, when data is deemed as "smooth"

*Kindly note:* If you mention cuSZ in your paper, please cite using [these](https://github.com/szcompressor/cuSZ#citing-cusz) BibTeX entries.


(C) 2020 by Washington State University and Argonne National Laboratory. See [COPYRIGHT](https://github.com/szcompressor/cuSZ/blob/master/LICENSE) in top-level directory.

- developers: Jiannan Tian, Cody Rivera, Wenyu Gai, Dingwen Tao, Sheng Di, Franck Cappello
- contributors (alphabetic): Jon Calhoun, Megan Hickman Fulp, Xin Liang, Robert Underwood, Kai Zhao
- Special thanks to Dominique LaSalle (NVIDIA) for serving as Mentor in Argonne GPU Hackaton 2021!

<br/>
<details>
<summary>
<b>
Table of Contents
</b>
</summary>

- [set up](#set-up)
- [use](#use)
  - [synopsis](#synopsis)
  - [example](#example)
- [FAQ](#faq)
- [tested by our team](#tested-by-our-team)
- [citing cuSZ](#citing-cusz)
- [acknowledgements](#acknowledgements)

</details>
<br/>

# set up

Requirements:

- NVIDIA GPU: Pascal, Volta, Turing, Ampere; CUDA 11 onward
  - We are working on the compatibility of CUDA 10.
- cmake 3.18 onward; C++14 enabled compiler, GCC 7 onward
  - [Ninja build system](https://ninja-build.org) is recommended.

<details>
<summary>
More details about build tools
</summary>

- The table gives a quick view of toolchain compatibility; please also refer to [a more detailed document](./doc/testbed.md).
- more reference: 1) [CUDA compilers](https://gist.github.com/ax3l/9489132), 2) [CUDA architectures & gencode](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/). 

|      |     |      |      |      |      |      |      |      |      |      |      |
| ---- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| gcc  | 7.x | 7.x  | 7.x  | 7.x  | 7.x  | 7.x  | 7.x  | 7.x  |      |      |      |
|      |     | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  |      |
|      |     |      |      |      | 9.x  | 9.x  | 9.x  | 9.x  | 9.x  | 9.x  | 9.x  |
| CUDA | 9.2 | 10.0 | 10.1 | 10.2 | 11.0 | 11.1 | 11.2 | 11.3 | 11.4 | 11.5 | 11.6 |

</details>

<br/>

```bash
git clone https://github.com/szcompressor/cuSZ.git cusz-latest
cd cusz-latest && chmod +x ./build.py
./build.py -t TARGET [-T BUILD TYPE] [-b BACKEND]
```

For example, to build A100-specific binary: `./build.py -t a100`

- Targets include 
  - `a100`, `v100`, `p100`, `ampere`, `turing`, `pascal`
  - `compat` for the maximum compatibility
- Build types include `release` (default), `release-profile` (with `-lineinfo`) and `debug` (with `-G`).
- Backends include `make` (default) and `ninja`
- `build.py` installs `cusz` binary to `${CMAKE_SOURCE_DIR}/bin`.
- `--purge` to clean up all the old builds.

<!-- Caveat: CUDA 10 or earlier, `cub` of a historical version becomes a dependency. After `git clone`, please use `git submodule update --init` to patch. -->


<br/>

# use

Type `cusz` or `cusz -h` for instant instructions. 

## synopsis

`<...>` for the required; `[...]` for the optional printout

```bash
# compression (-z)
cusz -t <type> -m <mode> -e <error bound> -i <file> -l <N-D size> -z [--report time]
# decompression (-x)
cusz -i <.cusza file> -x [--compare <original file>] [--report time]
```

## example 
```bash
export PATH=$(pwd)/bin:$PATH               ## specify the path temporarily
cd data && sh ./sh.get-sample-data         ## download sample data
CESM=$(pwd)/cesm-CLDHGH-3600x1800 EB=1e-4
cusz -t f32 -m r2r -e ${EB} -i ${CESM} -l 3600x1800 -z --report time
cusz -i ${CESM}.cusza -x --compare ${CESM} --report time
```

<!-- We use 1800-by-3600 (y-x order) CESM-ATM CLDHGH for demonstration, which is in preset and `-D cesm` can be alternatively used . Type `cusz` or `cusz -h` to look up the presets. -->

<!-- ```bash
cusz -t f32 -m r2r -e 1e-4 -i ./data/cesm-CLDHGH-3600x1800 --demo cesm -z
``` -->

The following *essential* arguments are required,

- `-z` to compress; `-x` to decompress.
- `-m` to specify error control mode from `abs` (absolute) and `r2r` (relative to value range)
- `-e` to specify error bound
- `-i` to specify input file
- `-l <size>` to specify dimensions

## advanced use

<details>
<summary>
skip writing decompressed data
</summary>

For evaluating purpose, we can skip writing to disk in decompression with `--skip write2disk`.

</details>

<!-- <details>
<summary>
additional lossless compression
</summary>

```bash
cusz -t f32 -m r2r -e 1e-4 -i ./data/cesm-CLDHGH-3600x1800 -l 3600,1800 -z --gzip
```
</details> -->

<details>
<summary>
specify output path
</summary>

```bash
mkdir data2 data3
# output compressed data to `data2`
cusz -t f32 -m r2r -e 1e-4 -i ./data/cesm-CLDHGH-3600x1800 -l 3600x1800 -z --opath data2
# output decompressed data to `data3`
cusz -i ./data2/cesm-CLDHGH-3600x1800.cusza -x --opath data3
```

</details>


<details>
<summary>
dryrun to learn data quality
</summary>

The actual compression or decompression is skipped; use `-r` or `--dry-run` in the command line.

```bash
# This works equivalently to decompress with `--origin /path/to/origin-datum`
cusz -t f32 -m r2r -e 1e-4 -i ./data/cesm-CLDHGH-3600x1800 -l 3600x1800 -r
```

</details>
<br/>

# FAQ

`*` There are implementation differences between CPU-SZ and cuSZ. Some technical detail is simplified. Please refer to our academic papers for more information.  

<details>
<summary>
How does SZ/cuSZ work?
</summary>

Prediction-based SZ algorithm comprises four major parts,

0. User specifies error-mode (e.g., absolute value (`abs`), or relative to data value magnitude (`r2r`) and error-bound.
1. Prediction errors are quantized in units of input error-bound (*quant-code*). Range-limited quant-codes are stored, whereas the out-of-range codes are otherwise gathered as *outlier*.
3. The in-range quant-codes are fed into Huffman encoder. A Huffman symbol may be represented in multiple bytes.
4. (CPU-only) additional DEFLATE method is applied to exploit repeated patterns. As of CLUSTER '21 cuSZ+ work, an RLE method performs a similar pattern-exploiting.

</details>

<details>
<summary>
Why is there multibyte symbol for Huffman?
</summary>

The principle of Huffman coding is to guarantee high-frequency symbols with fewer bits. To be specific, given arbitrary pairs of (symbol, frequency)-s, (*s<sub>i</sub>*, *f<sub>i</sub>*) and 
(*s<sub>j</sub>*, *f<sub>j</sub>*), the assigned codeword *c<sub>i</sub>* and *c<sub>j</sub>*, respectively, are guaranteed to have len(*c<sub>i</sub>*) is no greater than len(*c<sub>j</sub>*) if *f<sub>i</sub>* is no less than *f<sub>j</sub>*.

The combination of *n* single-byte does not reflect that quant-code representing the `+/-1` error-bound should be of the highest frequency. For example, an enumeration with 1024 symbols can cover 99.8% error-control code (the rest 0.2% can result in much more bits in codewords), among which the most frequent symbol can dominate at over 90%. If single-byte symbols are used, `0x00` from bytes near MSB makes

1. the highest frequency not properly represented, and
2. the pattern-exploiting harder. For example, `0x00ff,0x00ff` is otherwise interpreted as `0x00,0xff,0x00,0xff`.  

</details>

<details>
<summary>
What differentiates CPU-SZ and cuSZ in data quality and compression ratio?
</summary>

CPU-SZ offers a rich set of compression features and is far more mature than cuSZ. (1) CPU-SZ has preprocessing, more compression mode (e.g., point-wise), and autotuning. (2) CPU-SZ has Lorenzo predictor and Linear Regression, whereas cuSZ has Lorenzo (we are working on new predictors).

1. They share the same Lorenzo predictor. However, many factors affect data quality,
   1. preprocessing such as log transform and point-wise transform
   2. PSNR as a goal to autotune eb
   3. initial values from which we predict border values (as if padding). cuSZ predicts from zeros while SZ determines optimal values for, e.g., application-specific metrics. Also, note that cuSZ compression can result in a significantly higher PSNR than SZ (with the same eb, see Table 8 on page 10 of PACT '20 paper), but it is not necessarily better when it comes to applications.
   4. The PSNR serves as a generic metric: SZ guarantees a lower bound of PSNR when the eb is relative to the data range, e.g., 64 for 1e-3, 84 for 1e-4.
2. The linear scaling can be the same. SZ has an extra optimizer to decide the linear scaling range $[-r, +r]$; out-of-range quantization values are outliers. This is to optimize the compression ratio.
3. Currently, the Huffman encoding is the same except cuSZ partitions data (therefore, it has overhead in padding bits and partitioning metadata).

|        | preprocess | Lorenzo predictor | other predictors | Huffman | DEFLATE     |
| ------ | ---------- | ----------------- | ---------------- | ------- | ----------- |
| CPU-SZ | x          | x                 | x                | x       | x           |
| cuSZ   | TBD        | x, dual-quant     | TBD              | x       | alternative |
</details>

<details>
<summary>
What is cuSZ+?
</summary>

cuSZ+ is a follow-up peer-reviewed work in 2021, on top of the original 2020 work.
cuSZ+ mixes the improvements in decompression throughput (by 4.3x to 18.6x) and the use of data patterns that are the source of compressibility. 
There will not be, however, standalone software or version for cuSZ+. Instead, we are gradually rolling out the production-ready functionality mentioned in the published paper.

</details>

<details>
<summary>
What is the future plan/road map of cuSZ?
</summary>

1. more predictors based on domain-specific study and generality
2. more compression mode
3. both more modularized and more tightly coupled in components
4. APIs (soon)

</details>

<!-- <details>
<summary>
How to know compression ratio?
</summary>

The archive size is compress-time known. The archive includes metadata. -->

</details>

<details>
<summary>
How to know the performance?
</summary>

1. prior to CUDA 11: `nvprof <cusz command>`
2. CUDA 11 onward: `nsys profile --stat true <cusz command>`
3. enable `--report time` in CLI
4. A sample benchmark is shown at [`doc/benchmark.md`](https://github.com/szcompressor/cuSZ/blob/master/doc/benchmark.md). To be updated.

</details>
<br/>

# tested by our team

We tested cuSZ using datasets from [Scientific Data Reduction Benchmarks](https://sdrbench.github.io/) (SDRBench).

<details>
<summary>
datasets
</summary>

| dataset                                                                 | dim. | description                                                  |
| ----------------------------------------------------------------------- | ---- | ------------------------------------------------------------ |
| [EXAALT](https://gitlab.com/exaalt/exaalt/-/wikis/home)                 | 1D   | molecular dynamics simulation                                |
| [HACC](https://www.alcf.anl.gov/files/theta_2017_workshop_heitmann.pdf) | 1D   | cosmology: particle simulation                               |
| [CESM-ATM](https://www.cesm.ucar.edu)                                   | 2D   | climate simulation                                           |
| [EXAFEL](https://lcls.slac.stanford.edu/exafel)                         | 2D   | images from the LCLS instrument                              |
| [Hurricane ISABEL](http://vis.computer.org/vis2004contest/data.html)    | 3D   | weather simulation                                           |
| [NYX](https://amrex-astro.github.io/Nyx/)                               | 3D   | adaptive mesh hydrodynamics + N-body cosmological simulation |

We provide three small sample data in `data` by executing the script there. To download more SDRBench datasets, please use [`script/sh.download-sdrb-data`](script/sh.download-sdrb-data). 

</details>

<details>
<summary>
limitations
</summary>

- We are working on `double` support.
- We are working on integrating faster Huffman codec
- 4-byte Huffman symbol may break; `--config huffbyte=8` is needed.
- tuning performance regarding different data input size
- adding preprocessing (e.g., binning, log-transform, normalization)

</details>

<br/>

# citing cuSZ

PACT '20, cuSZ

```bibtex
@inproceedings{cusz2020,
      title = {cuSZ: An Efficient GPU-Based Error-Bounded Lossy Compression Framework for Scientific Data},
     author = {Tian, Jiannan and Di, Sheng and Zhao, Kai and Rivera, Cody and Fulp, Megan Hickman and Underwood, Robert and Jin, Sian and Liang, Xin and Calhoun, Jon and Tao, Dingwen and Cappello, Franck},
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

CLUSTER '21, cuSZ+

```bibtex
@INPROCEEDINGS {cuszplus2021,
      title = {Optimizing Error-Bounded Lossy Compression for Scientific Data on GPUs},
     author = {Tian, Jiannan and Di, Sheng and Yu, Xiaodong and Rivera, Cody and Zhao, Kai and Jin, Sian and Feng, Yunhe and Liang, Xin and Tao, Dingwen and Cappello, Franck},
       year = {2021},
      month = {September},
  publisher = {IEEE Computer Society},
    address = {Los Alamitos, CA, USA},
        url = {https://doi.ieeecomputersociety.org/10.1109/Cluster48925.2021.00047},
        doi = {10.1109/Cluster48925.2021.00047},
  booktitle = {2021 IEEE International Conference on Cluster Computing (CLUSTER)},
      pages = {283-293},
   keywords = {conferences;graphics processing units;computer architecture;cluster computing;reconstruction algorithms;throughput;encoding}
}
```

<br/>

# acknowledgements

This R&D is supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem. This repository is based upon work supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357, and also supported by the National Science Foundation under Grants [CCF-1617488](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1617488), [CCF-1619253](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1619253), [OAC-2003709](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2003709), [OAC-1948447/2034169](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2034169), and [OAC-2003624/2042084](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2042084).

![acknowledgement](https://user-images.githubusercontent.com/5705572/93790911-6abd5980-fbe8-11ea-9c8d-c259260c6295.jpg)
