#!/usr/bin/env python3
import os
import sys
import subprocess as sp

__author__ = "Jiannan Tian"
__copyright__ = "(C) 2021 by Washington State University, Argonne National Laboratory"
__license__ = "BSD 3-Clause"
__version__ = "0.3"
__date__ = "2021-07-15"

pascal = ["60", "61", "62"]
volta = ["70", "72"]
turing = ["75"]
ampere = ["80", "86"]

p100 = ["60"]
v100 = ["70"]
a100 = ["80"]


def figure_out_compatibility(cuda_ver):
    if cuda_ver == "":
        return ""

    max_compat = []
    if int(cuda_ver.replace(".", "")) < 92:
        print("Error")
    elif cuda_ver == "9.2":
        max_compat += pascal + volta
    elif cuda_ver in ["10.0", "10.1", "10.2"]:
        max_compat += pascal + volta + turing
    elif cuda_ver in ["11.0", "11.1", "11.2", "11.3", "11.4"]:
        max_compat += pascal + volta + turing + ampere
    else:
        max_compat += pascal + volta + turing + ampere
    return " ".join(max_compat)


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
    cuda_ver = (
        sp.check_output("""nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1""", shell=True)
        .decode("UTF-8")
        .strip()
    )
except:
    pass

targets = {
    "pascal": " ".join(pascal),
    "turing": " ".join(turing),
    "ampere": " ".join(ampere),
    "p100": " ".join(p100),
    "v100": " ".join(v100),
    "a100": " ".join(a100),
    "compat": figure_out_compatibility(cuda_ver),
}

types = {"release": "Release", "release-profile": "RelWithDebInfo", "debug": "Debug"}

if __name__ == "__main__":
    argc, argv = len(sys.argv), sys.argv

    if argc < 2:
        print(doc)
        exit(1)

    cmake_cmd = ""
    target, build_type = "", "Release"

    if argc == 2:  # modify target
        target = argv[1]

        if target not in targets.keys():
            print(doc)
            exit(1)

    if argc == 3:
        target = argv[1]
        build_type = argv[2]

        if (target in targets.keys()) and (build_type in types.keys()):
            pass
        else:
            print(doc)
            exit(1)

    cmake_cmd = 'cmake -DCMAKE_CUDA_ARCHITECTURES="{0}" -DCMAKE_BUILD_TYPE={1} -B {1}'.format(
        targets[target], build_type
    )

    # compile cusz
    if cuda_ver == "":
        print("No nvcc is found, stop compiling.", sys.stderr)
        exit(1)
    else:
        os.system(cmake_cmd)
        os.system("cd {0} && make -j".format(build_type))
        print("copying binary to ./bin/")
        if not os.path.isdir("./bin"):
            os.mkdir("./bin")
        os.system("cp {0}/cusz ./bin/cusz".format(build_type))
