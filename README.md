<h3 align="center"><img src="https://user-images.githubusercontent.com/10354752/81179956-05860600-8f70-11ea-8b01-856f29b9e8b2.jpg" width="150"></h3>

<h3 align="center">
A CUDA-Based Error-Bounded Lossy Compressor for Scientific Data
</h3>

<p align="center">
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
</p>

cuSZ is a CUDA implementation of the widely used [SZ lossy compressor](https://github.com/szcompressor/SZ) for scientific data. It is the *first* error-bounded lossy compressor (circa 2020) on GPU for scientific data, aming to massively improve SZ's throughput on heterogeneous HPC systems. 


(C) 2022 by Indiana University and Argonne National Laboratory. See [COPYRIGHT](https://github.com/szcompressor/cuSZ/blob/master/LICENSE) in top-level directory.

- developers: Jiannan Tian, Cody Rivera, Wenyu Gai, Dingwen Tao, Sheng Di, Franck Cappello
- contributors (alphabetic): Jon Calhoun, Megan Hickman Fulp, Xin Liang, Robert Underwood, Kai Zhao
- Special thanks to Dominique LaSalle (NVIDIA) for serving as Mentor in Argonne GPU Hackaton 2021!

<br>

<p align="center", style="font-size: 2em">
<a href="https://github.com/szcompressor/cuSZ/wiki/Build-and-Install"><b>build from source code</b></a>
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
<a href="https://github.com/szcompressor/cuSZ/wiki/Use"><b>use as a command-line tool</b></a>
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
<a href="https://github.com/szcompressor/cuSZ/wiki/API"><b>API reference</b></a>
</p>

<br>

<p align="center">
Kindly note: If you mention cuSZ in your paper, please cite using <a href="https://github.com/szcompressor/cuSZ/wiki/cite-our-works">these BibTeX entries</a>.
</p>


# FAQ

There are technical differences between CPU-SZ and cuSZ, please refer to our academic papers for more information.  

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

|        | preprocess | Lorenzo predictor | other predictors | Huffman | gzip/zstd   |
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


<details>
<summary>
How to know the performance?
</summary>

1. `nvprof <cusz command>` for GPUs prior to Ampere
2. `nsys profile --stat true <cusz command>` for all GPUs
3. enable `--report time` in CLI
4. A sample benchmark is shown at [`doc/benchmark.md`](https://github.com/szcompressor/cuSZ/blob/master/doc/benchmark.md). To be updated.

</details>


<details>
<summary>
What datasets are used?
</summary>

We tested cuSZ using datasets from [Scientific Data Reduction Benchmarks](https://sdrbench.github.io/) (SDRBench).

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
What are the limitations from the development?
</summary>

- The `double` support will be released in the next version.
- We are working on integrating faster Huffman codec
- 4-byte Huffman symbol may break; `--config huffbyte=8` is needed.
- tuning performance regarding different data input size
- adding preprocessing (e.g., binning, log-transform, normalization)

</details>

<br/>

# citing cuSZ

Our published papers cover the essential design and implementation. If you mention cuSZ in your paper, please cite using the BibTeX entries below.

**PACT '20: cuSZ** ([local copy](doc/PACT'20-cusz.pdf), [via ACM](https://dl.acm.org/doi/10.1145/3410463.3414624), or [via arXiv](https://arxiv.org/abs/2007.09625)) covers
  - framework: (fine-grained) *N*-D prediction-based error-controling "construction" + (coarse-grained) lossless encoding



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

- **CLUSTER '21: cuSZ+** ([local copy](doc/CLUSTER'21-cusz+.pdf) or [via IEEEXplore](https://doi.ieeecomputersociety.org/10.1109/Cluster48925.2021.00047})) covers
  - optimization in throughput, featuring fine-grained *N*-D "reconstruction"
  - optimization in compression ratio, when data is deemed as "smooth"

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

![acknowledgement](https://user-images.githubusercontent.com/10354752/196348936-f0909251-1c2f-4c53-b599-08642dcc2089.png)
