## DEMO for CUSZ::PREDICTION & NVCOMP::LOSSLESS 

This demo is a new compression pipeline, i.e., cuSZ's prediction and nvCOMP's lossless encoding (e.g., LZ4). Since this demo comes as an example in cuSZ, building it requires a complete build of cuSZ. nvCOMP now is distributed in binary, thus, specifying `/path/to/nvcomp` is required when building cuSZ.

### BUILD CUSZ WITH NVCOMP 

```bash
git clone https://github.com/szcompressor/cuSZ.git cusz-canary -b 221027-canary 
cd cusz-canary && mkdir build && cd build

# Example architectures (";"-separated when specifying)
#   Volta        : 70
#   Turing       : 75
#   Ampere       : 80 86
#   Ada Lovelace : 89 (as of CUDA 11.8)
#   Hopper       : 90 (as of CUDA 11.8)
cmake .. -DCUSZ_BUILD_EXAMPLES=on \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUSZ_BUILD_TESTS=on \
    -DCMAKE_PREFIX_PATH=[/path/to/nvcomp] \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86"
make -j
```

### DEMO

The source code of this demo locates at `cusz-latest/example/src/ck2.cu`.

```bash
## 2: "F" for fp32 type, "D" for fp64 type
## 3-5: [X] [Y] [Z] to specify dimentions
##      1D: "[X] 1 1"; 2D: "[X] [Y] 1"; 3D: "[X] [Y] [Z]"
## 6: absolute error bound, e.g, 1e-4
## 7: "ui16" (uint16_t) is intended for multibyte error-quantization code.
## 8: radius; 2x radius is used as the Huffman codebook size in cuSZ.
##             1         2        3   4   5   6            7    8
./example/ck2 [filename] [DTYPE] [X] [Y] [Z] [error-bound] ui16 512
```

### EXAMPLE DATASET

The example dataset can be downloaded from [SDRBench](https://sdrbench.github.io/). To download 2D 3600-by-1800 CESM-ATM data, 
```bash
wget https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/CESM-ATM/SDRBENCH-CESM-ATM-1800x3600.tar.gz

tar -zxvf SDRBENCH-CESM-ATM-1800x3600.tar.gz
```

### RUN THE DEMO: PREDICTION + NVCOMP::LZ4

```bash
## in `build` dir
export CESM=/path/to/1800x3600/CLDHGH_v_1_1800_3600.f32
./example/ck2 ${CESM} F 3600 1800 1 1e-4 ui16 512

## a sample decompression data quality is listed below

#                      data-len        data-byte         fp-type?                 
#                       6480000                4              yes
#                           min              max              rng              std
#   origin        3.3749157e-07       0.92075449       0.92075413       0.19757324
#   eb-lossy                  0       0.92079997       0.92079997       0.19762968
#                       abs-val          abs-idx           pw-rel           VS-RNG
#   max-error     0.00010001659          1974554              nan    0.00010862465
#                            CR            NRMSE        cross-cor             PSNR
#   metrics           2.9544599    6.2703313e-05        0.9999999         84.05419
#                      lag1-cor         lag2-cor                                  
#   auto               0.999838         0.999464
```


### RUN THE ORIGINAL CUSZ
```bash
## in `build` dir; type `cusz` to get help
## running with the same configuration
./cusz -t f32 -m abs -e 1e-4 -i ${CESM} -l 3600x1800 -z --report time
./cusz -i ${CESM}.cusza -x --report time --compare ${CESM}

## a sample decompression data quality is listed below

#                      data-len        data-byte         fp-type?                 
#                       6480000                4              yes
#                           min              max              rng              std
#   origin        3.3749157e-07       0.92075449       0.92075413       0.19757324
#   eb-lossy                  0       0.92079997       0.92079997       0.19762968
#                       abs-val          abs-idx           pw-rel           VS-RNG
#   max-error     0.00010001659          1974554              nan    0.00010862465
#                            CR            NRMSE        cross-cor             PSNR
#   metrics           6.1702592    6.2703313e-05        0.9999999         84.05419
#                      lag1-cor         lag2-cor                                  
#   auto               0.999838         0.999464
```

Note that with `uint16` error-quantization code (reinterpreted as two `uint8`) to feed `lz4`, the compression ratio (CR) of the new compression pipeline (cusz::prediction and nvcomp::lz4) is not optimal. In the above test, CR of original cuSZ is 6.1702592, whereas CR of the new compression pipeline is 2.9544599 (even ignoring prediction outliers).
