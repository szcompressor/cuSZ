# Verified Testbed & Toolchain

Our software supports GPUs with the following compute capabilities:
- Pascal; `sm_60`, `sm_61`
- Volta; `sm_70`
- Turing; `sm_75`
- Ampere; `sm_80`, `sm_86`

## The Tested 
The toolchain combination is non-exhaustible. We listed our experienced testbed below. Number in GPU-(GCC,CUDA) combination denotes major version of compiler ever tested.

| setup     | arch.  | SM  |     |      |      |      |      |      |      |      |      |      |      |
| --------- | ------ | --- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| GCC       |        |     | 7.x | 7.x  | 7.x  | 7.x  | 7.x  | 7.x  | 7.x  | 7.x  |      |      |      |
|           |        |     |     | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  | 8.x  |      |      |      |
|           |        |     |     |      |      |      | 9.x  | 9.x  | 9.x  | 9.x  | 9.x  | 9.x  | 9.x  |
| CUDA      |        |     | 9.2 | 10.0 | 10.1 | 10.2 | 11.0 | 11.1 | 11.2 | 11.3 | 11.4 | 11.5 | 11.6 |
|           |        |     |     |      |      |      |      |      |      |      |      |      |      |
| P2000M    | Pascal | 61  |     |      |      |      | 7    |      |      |      |      |      |      |
| RTX 2060S | Turing | 75  | 7   | 7    | 7    | 7    | 9    | 9    | 9    | 9    | 9    | 9    | 9    |
| RTX 5000  | Turing | 75  |     |      | 7/8  |      |      |      |      |      |      |      |      |
| RTX 8000  | Turing | 75  |     |      | 7    |      |      |      |      |      |      |      |      |
| RTX 3080  | Ampere | 86  |     |      | 7    |      |      | 9    | 9    |      |      |      |      |
|           |        |     |     |      |      |      |      |      |      |      |      |      |      |
| P100      | Pascal | 60  |     |      | 7    |      |      |      |      |      |      |      |      |
| V100      | Volta  | 70  | 7   |      |      | 7/8  |      | 7    |      |      |      |      |      |
| A100      | Ampere | 80  |     |      |      |      | 7-9  | 7-9  | 7-9  |      | 9    | 9    | 9    |



## Ubuntu 20.04 LTS

It comes with GCC 9.3 and is confirmed workin giwth any CUDA 11.x: no extra setup is needed.

## Fedora 35
- CUDA 11.6 + GCC 11.2 confirmed **not** working (as of 2022 Feburuary).
- Because of `<functional>` header inclusion error, even though it is [officially supported](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements).
- Others experienced the similar issue: [case 1](https://github.com/NVlabs/instant-ngp/issues/119), [case 2](https://forums.developer.nvidia.com/t/cuda-11-6-0-with-gcc-11-2-1-fails-to-process-system-headers-included-by-functional/203556) 

Fedora only provides single-version GCC, i.e., Fedora 35 + GCC 11.2. A workaround is use 3rd-party pacakge manager to install other version of GCC, e.g., [linuxbrew](https://brew.sh), [spack](https://spack.io). A linuxbrew setup is given below 

```bash
# defult path to install from .run file
export CUDA_ROOT=/usr/local/cuda  

# install gcc
brew install gcc@9

# get the absolute paths
sudo ln -s /home/linuxbrew/.linuxbrew/bin/gcc-9 $CUDA_ROOT/bin/gcc
sudo ln -s /home/linuxbrew/.linuxbrew/bin/g++-9 $CUDA_ROOT/bin/g++

# consider to put these in ~/.zshrc or ~/.bashrc
export CC=gcc-9
export CXX=g++-9
```

