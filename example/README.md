### BUILD

```bash
git clone https://github.com/boyuanzhang62/cuSZ.git cusz-latest
git switch canary-by
cd cusz-latest && mkdir build && cd build

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

### RUN

The source file is in `cusz-latest/example/src/ck2.cu`

```bash
# Use ./example/ck2 to get help information
# Here is an example
./example/ck2 [/dir/to/file] F [X] [Y] [Z] [error-bound] ui32 128
```
