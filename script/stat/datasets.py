__author__ = "Jiannan Tian"
__copyright__ = "(C) 2021 by Washington State University, Argonne National Laboratory"
__license__ = "BSD 3-Clause"
__version__ = "0.2.x"
__date__ = "2021-03-28"


class Constant(object):
    def __init__(self, **kwargs):
        self.x = 1
        self.y = 1
        self.z = 1
        self.ndim = 1

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.stride1 = self.x
        self.stride2 = self.y * self.x
        self.ld = self.x  # if 2D
        self.size = self.x * self.y * self.z


class Block(object):
    def __init__(self, constant: Constant):
        self.side = block_sides[constant.ndim]
        self.length = self.side**constant.ndim
        self.nx = (constant.x - 1) // self.side + 1
        self.ny = (constant.y - 1) // self.side + 1
        self.nz = (constant.z - 1) // self.side + 1
        self.stride1 = self.side
        self.stride2 = self.side**2
        self.total_num = self.nz * self.ny * self.nx


block_sides = {1: 256, 2: 16, 3: 8}
hacc = {"x": 3600, "y": 1, "z": 1, "ndim": 1}
cesm = {"x": 3600, "y": 1800, "z": 1, "ndim": 2}
hurricane = {"x": 500, "y": 500, "z": 100, "ndim": 3}
nyx_small = {"x": 512, "y": 512, "z": 512, "ndim": 3}
nyx_medium = {"x": 1024, "y": 1024, "z": 1024, "ndim": 3}
qmcpack = {"x": 3600, "y": 1800, "z": 1, "ndim": 3}
