cuSZ: A GPU Accelerated Error-Bounded Lossy Compressor
---

# `changelog`
May, 2020
- add `--skip huffman` and `--verify huffman` feature
- add binning support (merge from `exafel` branch)

April, 2020
- use `cuSparse` to deflate outlier array
- add `argparse` to check and parse argument inputs
- add CUDA wrappers (e.g., `mem::CreateCUDASpace`)
- add concise and detailed help doc

# Compile

```bash
git clone git@github.com:hipdac-lab/cuSZ.git
cd cuSZ
```

```bash
cd src
make cusz     # compile cusz for {1,2,3}-D, with Huffman codec
```

# Run
- `cusz` or `cusz -h` for detailed instruction

## Dual-Quantization with Huffman
- To run a demo
```bash
./cusz
    -f32                # specify data type
    -m <abs|r2r>	# absolute error bound mode OR value-range-based relative error bound mode
    -e <num>		# error bound
    -D <demo dataset>
    -z -x		# for demonstration purpose, you can do both zip (compresss) and extract (decompress) in one execution
    -i <input data>	# input data path
```
and to execute
```bash
./cusz -f32 -m r2r -e 1.23e-4.56 -D cesm -i CLDHGH_sample -z -x
```
- We provide a dry-run mode to quickly get the summary of compression quality (without Huffman coding and decompression)
```bash
./cusz -f32 -m r2r -e 1.23e-4.56 -D cesm -i CLDHGH_sample -r
```
- To run cuSZ on any given data field with arbitrary input size(s)
```bash
./cusz
    -f32                # specify datatype
    -m <abs|r2r>	# absolute mode OR relative to value range
    -e <num>		# error bound
    -z -x		# for demonstration, we both archive (compresss) and extract (decompress)
    -i <input data>	# input data path
    -2 nx ny		# dimension(s) with low-to-high order
```
and to execute
```bash
./cusz -f32 -m r2r -e 1.23e-4.56 -i CLDHGH_sample -2 3600 1800 -z -x
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

## Reference
 - [1] Gómez-Luna, Juan, José María González-Linares, José Ignacio Benavides, and Nicolás Guil. "An optimized approach to histogram computation on GPU." Machine Vision and Applications 24, no. 5 (2013): 899-908.
 - [2] Barnett, Mark L. "Canonical Huffman encoded data decompression algorithm." U.S. Patent 6,657,569, issued December 2, 2003.
