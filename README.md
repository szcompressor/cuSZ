<h3 align="center"><img src="https://user-images.githubusercontent.com/10354752/81179956-05860600-8f70-11ea-8b01-856f29b9e8b2.jpg" width="150"></h3>

<h3 align="center">
A CUDA-Based Error-Bounded Lossy Compressor for Scientific Data
</h3>

<p align="center">
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
</p>

cuSZ is a CUDA implementation of the widely used [SZ lossy compressor](https://github.com/szcompressor/SZ) for scientific data. It is the *first* error-bounded lossy compressor (circa 2020) on GPU for scientific data, aming to massively improve SZ's throughput on heterogeneous HPC systems. 


(C) 2022 by Indiana University and Argonne National Laboratory. See [COPYRIGHT](https://github.com/szcompressor/cuSZ/blob/master/LICENSE) in top-level directory.

- developers: (framework) Jiannan Tian, (kernel/pipeline) Jinyang Liu, Shixun Wu, Cody Rivera, (deployment) Robert Underwood, (advisors, PIs) Dingwen Tao, Sheng Di, Franck Cappello
- contributors (alphabetic): Jon Calhoun, Wenyu Gai, Megan Hickman Fulp, Xin Liang, Kai Zhao
- Special thanks to Dominique LaSalle (NVIDIA) for serving as Mentor in Argonne GPU Hackaton 2021!

<br>

<p align="center", style="font-size: 2em">
<a href="https://github.com/szcompressor/cuSZ/wiki/Build-and-Install"><b>build from source code</b></a>
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
<a href="https://github.com/szcompressor/cuSZ/wiki/Use"><b>use as a command-line tool</b></a>
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
<a href="https://github.com/szcompressor/cuSZ/wiki/API"><b>API reference</b></a>
</p>

<!-- <br> -->

<p align="center">
Kindly note: If you mention cuSZ in your paper, please refer to <a href="#citing-cusz">the detail below</a>.
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

</details>

<details>
<summary>
How does cuSZ evolve over years?
</summary>

cuSZ and its variants use variable techniques to balance the need for data-reconstruction quality, compression ratio, and data-processing speed. A quick comparison is given below.

Notably, cuSZ (Tian et al., '20, '21) as the basic framework provides a balanced compression ratio and quality, while FZ-GPU (Zhang, Tian et al., '23) and SZp-CUDA/GSZ (Huang et al., '23, '24) prioritize data processing speed. cuSZ+ (hi-ratio) is an outcome of data compressibility research to demonstrate that certain methods (e.g., RLE) can work better in highly compressible cases (Tian et al., '21). The latest art, cuSZ-i (Liu, Tian, Wu et al., '24), attempts to utilize the QoZ-like methods (Liu et al., '22) to significantly enhance the data-reconstruction quality and the compression ratio.

```
                    prediction &                 statistics          lossless encoding          lossless encoding
                    quantization                                     passs (1)                  pass (2)

                  +----------------------+      +-----------+      +------------------+       +-----------------+
CPU-SZ     -----> | predictor {ℓ, lr, S} | ---> | histogram | ---> | ui2 Huffman enc. | ----> | DEFLATE (LZ+HF) |
'16, '17-ℓ, '18-lr, '21-S, '22-QoZ ------+      +-----------+      +------------------+       +-----------------+
(Di and Franck, Tao et al., Liang et al. Zhao et al., Liu et al.)

                  +----------------------+      +-----------+      +------------------+
cuSZ       -----> | predictor ℓ-(1,2,3)D | ---> | histogram | ---> | ui2 Huffman enc. | ----> ( n/a )
'20, '21          +----------------------+      +-----------+      +------------------+
(Tian et al.)
                  +----------------------+      +-----------+      +-------------------+      +---------+
cuSZ+        ---> | predictor ℓ-(1,2,3)D | ---> | histogram | ---> | de-redundancy RLE | ---> | HF enc. |
hi-ratio '21      +----------------------+      +-----------+      +-------------------+      +---------+
(Tian et al.)
                  +----------------------+                         +---------------+
FZ-GPU '23   ---> | predictor ℓ-(1,2,3)D | ---> ( n/a ) ---------> | de-redundancy | -------> ( n/a )
(Zhang, Tian et al.) --------------------+                         +---------------+

                  [ single kernel ]------------------------------------------------+           
SZp-CUDA/GSZ ---> | predictor ℓ-1D   ---------> ( n/a ) --------->   de-redundancy | -------> ( n/a )
'23, '24          +----------------------------------------------------------------+           
(Huang et al.)

                  +----------------+            +-----------+      +------------------+       +---------------+
cuSZ-i '24   ---> | predictor S-3D | ---------> | histogram | ---> | ui2 Huffman enc. | ----> | de-redundancy |
(Liu, Tian, Wu et al.) ------------+            +-----------+      +------------------+       +---------------+

ℓ: Lorenzo predictor; lr: linear-regression predictor; S: spline-interpolative predictor
```

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


# cite cuSZ

Our published papers cover the essential design and implementation. If you mention cuSZ in your paper, please kindly cite using `\cite{tian2020cusz,tian2021cuszplus,liu_tian_wu2024cuszi}` and the BibTeX entries below (or standalone [`.bib` file](doc/psz-cusz.bib)).

1. **PACT '20: cuSZ** ( [local copy](doc/PACT'20-cusz.pdf) | [ACM](https://dl.acm.org/doi/10.1145/3410463.3414624) | [arXiv](https://arxiv.org/abs/2007.09625) ) covers
    - basic framework: (fine-grained) *N*-D prediction-based error-controling "construction" + (coarse-grained) lossless encoding
2. **CLUSTER '21: cuSZ+** ( [local copy](doc/CLUSTER'21-cusz+.pdf) | [IEEEXplore](https://doi.ieeecomputersociety.org/10.1109/Cluster48925.2021.00047}) | [arXiv](https://arxiv.org/abs/2105.12912) ) covers
    - optimization in throughput, featuring fine-grained *N*-D "reconstruction"
    - optimization in compression ratio, when data is deemed as "smooth"
3. **SC '24: cuSZ-_i_** (The final paper will come to SC '24 proceedings.) The paper ( [arXiv](https://arxiv.org/abs/2312.05492) ) covers
    - spline-interpolation-based high-ratio data compression and high-quality data reconstruction
    - compresion ratio boost from incorporating the synergetic lossless encoding

```bibtex
@inproceedings{tian2020cusz,
      title = {{{\textsc cuSZ}: An efficient GPU-based error-bounded lossy compression framework for scientific data}},
     author = {Tian, Jiannan and Di, Sheng and Zhao, Kai and Rivera, Cody and Fulp, Megan Hickman and Underwood, Robert and Jin, Sian and Liang, Xin and Calhoun, Jon and Tao, Dingwen and Cappello, Franck},
       year = {2020}, month = {10},
        doi = {10.1145/3410463.3414624}, isbn = {9781450380751},
  booktitle = {Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques},
     series = {PACT '20}, address = {Atlanta (virtual event), GA, USA}}

@inproceedings{tian2021cuszplus,
      title = {Optimizing error-bounded lossy compression for scientific data on GPUs},
     author = {Tian, Jiannan and Di, Sheng and Yu, Xiaodong and Rivera, Cody and Zhao, Kai and Jin, Sian and Feng, Yunhe and Liang, Xin and Tao, Dingwen and Cappello, Franck},
       year = {2021}, month = {09},
        doi = {10.1109/Cluster48925.2021.00047},
  booktitle = {2021 IEEE International Conference on Cluster Computing (CLUSTER)},
     series = {CLUSTER '21}, address = {Portland (virtual event), OR, USA}}

@article{liu_tian_wu2024cuszi,
     title = {{{\scshape cuSZ}-{\itshape i}: High-ratio scientific lossy compression on
             GPUs with optimized multi-level interpolation}},
    author = {Jinyang Liu and Jiannan Tian and Shixun Wu and Sheng Di and Boyuan Zhang and Yafan Huang and Kai Zhao and Guanpeng Li and Dingwen Tao and Zizhong Chen and Franck Cappello},
      year = {2024}, month = {11},
      note = {Co-first authors: Jinyang Liu, Jiannan Tian, and Shixun Wu},
       doi = {10.48550/arXiv.2312.05492},
    series = {SC '24}, address = {Atlanta, GA, USA}}
```


# acknowledgements

This R&D is supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem. This repository is based upon work supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357, and also supported by the National Science Foundation under Grants [CCF-1617488](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1617488), [CCF-1619253](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1619253), [OAC-2003709](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2003709), [OAC-1948447/2034169](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2034169), and [OAC-2003624/2042084](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2042084).

![acknowledgement](https://user-images.githubusercontent.com/10354752/196348936-f0909251-1c2f-4c53-b599-08642dcc2089.png)
