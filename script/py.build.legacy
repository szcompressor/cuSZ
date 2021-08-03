#!/usr/bin/env python3
import os
import sys
import subprocess as sp

__author__ = "Jiannan Tian"
__copyright__ = "(C) 2021 by Washington State University, Argonne National Laboratory"
__license__ = "BSD 3-Clause"
__version__ = "0.3"
__date__ = "2021-01-12"

# current support and backward compatibility; t=Tesla, cp=consumer,professional
SMp_PASCAL_t = "-gencode=arch=compute_60,code=sm_60"
SMp_PASCAL_cp = "-gencode=arch=compute_61,code=sm_61"
SMp_VOLTA = "-gencode=arch=compute_70,code=sm_70"
SMp_TURING = "-gencode=arch=compute_75,code=sm_75"
SMp_AMPERE_t = "-gencode=arch=compute_80,code=sm_80"
SMp_AMPERE_cp = "-gencode=arch=compute_86,code=sm_86"
# forward compatibility
compute_PASCAL_t = "-gencode=arch=compute_60,code=compute_60"
compute_PASCAL_cp = "-gencode=arch=compute_61,code=compute_61"
compute_VOLTA = "-gencode=arch=compute_70,code=compute_70"
compute_TURING = "-gencode=arch=compute_75,code=compute_75"
compute_AMPERE_t = "-gencode=arch=compute_80,code=compute_80"
compute_AMPERE_cp = "-gencode=arch=compute_86,code=compute_86"


def figure_out_compatibility(cuda_ver):
    if cuda_ver == "":
        return ""

    max_compat = ["-arch=sm_60"]
    if int(cuda_ver.replace(".", "")) < 92:
        print("Error")
    elif cuda_ver == "9.2":
        max_compat = [
            "-arch=sm_60", SMp_PASCAL_t, SMp_PASCAL_cp, SMp_VOLTA,
            compute_VOLTA
        ]
    elif cuda_ver in ["10.0", "10.1", "10.2"]:
        max_compat = [
            "-arch=sm_60", SMp_PASCAL_t, SMp_PASCAL_cp, SMp_VOLTA, SMp_TURING,
            compute_TURING
        ]
    elif cuda_ver in ["11.0", "11.1", "11.2", "11.3"]:
        max_compat = [
            "-arch=sm_60", SMp_PASCAL_t, SMp_PASCAL_cp, SMp_VOLTA, SMp_TURING,
            SMp_AMPERE_t, SMp_AMPERE_cp, compute_AMPERE_cp
        ]
    else:
        # CUDA 12 onward may change the lowest compatible sm
        max_compat = [
            "-arch=sm_60", SMp_PASCAL_t, SMp_PASCAL_cp, SMp_VOLTA, SMp_TURING,
            SMp_AMPERE_t, SMp_AMPERE_cp, compute_AMPERE_cp
        ]
    return " ".join(max_compat)


P100 = ["-arch=sm_60", SMp_PASCAL_t, compute_PASCAL_t]
V100 = ["-arch=sm_70", SMp_VOLTA, compute_VOLTA]
A100 = ["-arch=sm_80", SMp_AMPERE_t, compute_AMPERE_t]
Pascal = ["-arch=sm_61", SMp_PASCAL_cp, compute_PASCAL_cp]
Turing = ["-arch=sm_75", SMp_TURING, compute_TURING]
Ampere = ["-arch=sm_86", SMp_AMPERE_cp, compute_AMPERE_cp]

doc = """./build.py <gpu.name> <optional: build type>
example:    ./build.py turing
            ./build.py turing release

build.type: release  release-profile  debug
gpu.names:  p100  v100  a100
            pascal  turing  ampere
            compat
"""

cuda_ver = ""
try:
    cuda_ver = sp.check_output(
        """nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1""",
        shell=True).decode("UTF-8").strip()
except:
    pass

build_target = {
    "pascal": " ".join(Pascal),
    "turing": " ".join(Turing),
    "ampere": " ".join(Ampere),
    "p100": " ".join(P100),
    "v100": " ".join(V100),
    "a100": " ".join(A100),
    "compat": figure_out_compatibility(cuda_ver)
}

build_types = {
    "release": {
        "host": "-O3",
        "cuda": "-O2"
    },
    "release-profile": {
        "host": "-O3 -g",
        "cuda": "-O2 -pg -lineinfo"
    },
    "debug": {
        "host": "-O0 -g",
        "cuda": "-G"
    }
}

if __name__ == "__main__":
    with open("script/tpl.Makefile", "r") as fi:
        makefile = fi.read()
    argc, argv = len(sys.argv), sys.argv

    if argc < 2:
        print(doc)
        exit(1)

    rule, opt_level = "", ""

    if argc == 2:  # modify target
        if (argv[1]) not in build_target.keys():
            print(doc)
            exit(1)
        rule = argv[1]
        makefile = makefile.replace(  #
            "HW_TARGET += ####",  #
            "HW_TARGET += ####".replace("####", build_target[rule])  #
        )
        makefile = makefile.replace(  #
            "CC_FLAGS  += ####",  #
            "CC_FLAGS  += ####".replace("####",
                                        build_types["release"]["host"])  #
        )
        makefile = makefile.replace(  #
            "NV_FLAGS  += ####",  #
            "NV_FLAGS  += ####".replace("####",
                                        build_types["release"]["cuda"])  #
        )

    if argc == 3:
        if (argv[1] in build_target.keys()) and (argv[2] in build_types.keys()):
            rule, opt_level = argv[1], argv[2]
        elif (argv[2] in build_target.keys()) and (argv[1] in build_types.keys()):
            rule, opt_level = argv[2], argv[1]
        else:
            print(doc)
            exit(1)

        makefile = makefile.replace(  #
            "HW_TARGET += ####",  #
            "HW_TARGET += ####".replace("####", build_target[rule])  #
        )
        makefile = makefile.replace(  #
            "CC_FLAGS  += ####",  #
            "CC_FLAGS  += ####".replace("####",
                                        build_types[opt_level]["host"])  #
        )
        makefile = makefile.replace(  #
            "NV_FLAGS  += ####",  #
            "NV_FLAGS  += ####".replace("####",
                                        build_types[opt_level]["cuda"])  #
        )

    with open("Makefile", "w") as fo:
        fo.write(makefile)

    ############################################################
    # submodules
    ############################################################
    NVCOMP_DIR         = "external/nvcomp"
    NVCOMP_INCLUDE_DIR = NVCOMP_DIR + "/build/include"
    NVCOMP_LIB_DIR     = NVCOMP_DIR + "/build/lib"
    NVCOMP_STATIC_LIB  = NVCOMP_LIB_DIR + "/libnvcomp.a"

    GTEST_DIR         = "external/googletest"
    GTEST_INCLUDE_DIR = GTEST_DIR + "/googletest/include"
    GTEST_LIB_DIR     = GTEST_DIR + "/build/lib"
    GTEST_STATIC_LIB  = GTEST_LIB_DIR + "/libgtest.a"

    cc = sp.check_output("which gcc", shell=True).decode("utf-8").strip()
    cxx = sp.check_output("which g++", shell=True).decode("utf-8").strip()

    patch_nvcomp = "patch {0}/src/CMakeLists.txt external/patch.nvcomp-1.1".format(NVCOMP_DIR)
    cmake_nvcomp_pre_cuda11 = "cmake -DCUB_DIR=$(pwd)/external/cub -DCMAKE_C_COMPILER={1} -DCMAKE_CXX_COMPILER={2} -S {0} -B {0}/build  && make -j -C {0}/build".format(NVCOMP_DIR, cc, cxx)
    cmake_nvcomp_cuda11_onward = "cmake -DCMAKE_C_COMPILER={1} -DCMAKE_CXX_COMPILER={2} -S {0} -B {0}/build  && make -j -C {0}/build".format(NVCOMP_DIR, cc, cxx)
    cmake_nvcomp = cmake_nvcomp_cuda11_onward if cuda_ver in ["11.0", "11.1", "11.2", "11.3"] else cmake_nvcomp_pre_cuda11
    cmake_gtest  = "cmake -DCMAKE_C_COMPILER={1} -DCMAKE_CXX_COMPILER={2} -S {0} -B {0}/build  && make -j -C {0}/build""".format(GTEST_DIR, cc, cxx)

    # print(patch_nvcomp)
    # print(cmake_nvcomp_pre_cuda11)
    # print(cmake_nvcomp_cuda11_onward)
    # print(cmake_gtest)
    # exit()

    # TODO purge

    compile_info = ""

    ## disabled for modular design and refactoring

    # if not os.path.exists("external/nvcomp/build/lib/libnvcomp.a"):
    #     os.system(patch_nvcomp)
    #     os.system(cmake_nvcomp)
    # if not os.path.exists("external/nvcomp/build/lib/libgtest.a"):
    #     os.system(cmake_gtest)

    # # double check
    # if os.path.exists(GTEST_STATIC_LIB):
    #     print("gtest lib ready")
    # if os.path.exists(NVCOMP_STATIC_LIB):
    #     print("nvcomp lib ready")

    ############################################################
    # compile cusz
    ############################################################
    if cuda_ver == "":
        print("No nvcc is cound, skip compilation.")
        exit(1)
    else:
        os.system("make clean & make")
