#!/usr/bin/env python3
import os
import sys
import subprocess as sp
import argparse

__author__ = "Jiannan Tian"
__copyright__ = "(C) 2021 by Washington State University, Argonne National Laboratory"
__license__ = "BSD 3-Clause"
__version__ = "0.3"
__date__ = "2021-07-15"  ## rev.1 2022-01-13

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
    return ";".join(max_compat)


cuda_ver = ""
try:
    cuda_ver = (sp.check_output(
        """nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1""",
        shell=True).decode("UTF-8").strip())
except:
    pass

targets = {
    "pascal": ";".join(pascal),
    "turing": ";".join(turing),
    "ampere": ";".join(ampere),
    "p100": ";".join(p100),
    "v100": ";".join(v100),
    "a100": ";".join(a100),
    "compat": figure_out_compatibility(cuda_ver),
}

types = {
    "release": "Release",
    "release-profile": "RelWithDebInfo",
    "debug": "Debug"
}

parser = argparse.ArgumentParser(
    description='Build cuSZ compressor.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--backend',
                    '-b',
                    default='make',
                    help='build system: ninja, make')
parser.add_argument(
    '--target',
    '--gpu',
    '-t',
    default='turing',
    help='GPU targets include: a100 v100 p100; ampere turing pascal; compat')
parser.add_argument('--type',
                    '-T',
                    default='release',
                    help='release, debug, release-profile')
parser.add_argument('--purge',
                    action='store_true',
                    help='purge all contents in old builds')

argc, argv = len(sys.argv), sys.argv
args = parser.parse_args()


def build(cmake_cmd, build_cmd, build_type):
    # print(cmake_cmd)
    os.system(cmake_cmd)
    os.system(build_cmd)
    print("copying binary to ./bin/")
    if not os.path.isdir("./bin"):
        os.mkdir("./bin")
    os.system("cp {0}/cusz ./bin/cusz".format(build_type))


if argc < 2:
    parser.print_help()

if args.purge:
    purge_cmd = "rm -fr Release Debug RelWithDebInfo"
    print("purge old builds...")
    os.system(purge_cmd)
    exit(0)

build_type = types[args.type]

_arch = '-DCMAKE_CUDA_ARCHITECTURES="{0}"'.format(targets[args.target])
_type = '-DCMAKE_BUILD_TYPE={0}'.format(build_type)
_dir = '-B {0}'.format(build_type)
_ninja = '-GNinja'
_with_make = ["cmake", _arch, _type, _dir]
_with_ninja = ["cmake", _arch, _type, _dir, _ninja]
cmake_cmd = None
if args.backend == 'ninja':
    cmake_cmd = " ".join(_with_ninja)
else:
    cmake_cmd = " ".join(_with_make)
build_cmd = "cd {0} && ninja".format(
    build_type) if args.backend == 'ninja' else "cd {0} && make -j".format(
        build_type)

# compile cusz
if cuda_ver == "":
    print("No nvcc is found, stop compiling.", sys.stderr)
    exit(1)
else:
    if argc < 2:
        print('\nbuilding according to the default...')
    else:
        print('\nbuilding...')
    print("""
target  : {0} ({1})
type    : {2}
backend : {3}
        """.format(args.target.capitalize(), "SM " + targets[args.target],
                   build_type, args.backend.capitalize()))
    build(cmake_cmd, build_cmd, build_type)
