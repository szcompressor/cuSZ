# set up spack
```bash
## We are waiting for merge to upstream. For now, we are using
git clone https://github.com/dingwentao/spack.git   ## will create `spack` directory
```

Acccording to [Spack instruction](https://spack.readthedocs.io/en/latest/getting_started.html),
```bash
# For bash/zsh users
export SPACK_ROOT=/path/to/spack
export PATH=$SPACK_ROOT/bin:$PATH
```
And for better experience, we sugguest
```bash
# Note you must set SPACK_ROOT
# For bash/zsh users
$ . $SPACK_ROOT/share/spack/setup-env.sh
```

# install cuSZ
- CUDA 9.2+ with proper version of `gcc` is required; refer to [README](../README.md) in the top-level directory.
- If you are using spack on your personal computer or local workstation, with system default `gcc` that is no lower than 7.3.0 as your compiler, you can just try `spack install cusz`.
- If you are using spack on an HPC cluster with [Environment Modules](http://modules.sourceforge.net/) and [LMod](http://lmod.readthedocs.io/en/latest/) (i.e., use `module load` to activate your toolchain), you may not pass the compliation due to `gcc` + CUDA + `module load` exposed library path. As is pointed out by [Axel Huebl in here](https://picongpu.readthedocs.io/en/latest/install/instructions/spack.html), a workaround can be fully compile a `gcc-7.3.0` under the namespace of Spack. A full setup is shown below,
    ```bash
    spack install gcc@7.3.0 && spack load gcc@7.3.0 && spack compiler add
    spack install cusz %gcc@7.3.0
    ```
- After succefull installation, `spack load cusz` and it works out-of-box.