__author__ = "Jiannan Tian"
__copyright__ = "(C) 2021 by Washington State University, Argonne National Laboratory"
__license__ = "BSD 3-Clause"
__version__ = "0.2.x"
__date__ = "2021-03-29"
__change__ = "(rev) 2021-03-30"

import numpy as np
from typing import List, Union, Tuple
import sys
import pretty_errors
import datasets
from functools import reduce

pretty_errors.configure(separator_character='*',
                        filename_display=pretty_errors.FILENAME_EXTENDED,
                        line_number_first=True,
                        display_link=True,
                        lines_before=5,
                        lines_after=2,
                        line_color=pretty_errors.RED + '> ' +
                                   pretty_errors.default_config.line_color,
                        code_color='  ' +
                                   pretty_errors.default_config.line_color,
                        truncate_code=True,
                        display_locals=True)


# CESM example
# sample_name = "CLDLOW.dat"
# origin = np.fromfile(sample_name, dtype=np.float32).reshape(cesm.Constants.y, cesm.Constants.x)
# quant = np.fromfile(sample_name + ".quant1e-4", dtype=np.uint16).reshape(cesm.Constants.y, cesm.Constants.x)


class HuffEstStruct:
    def __init__(self):
        self.entropy: float = 0.0
        self.redundancy_lb: float = 0.0
        self.avg_bitlen_lb: float = 0.0
        self.cr_ub: float = 0.0

        self.redundancy_ub: float = 0.0
        self.avg_bitlen_ub: float = 0.0
        self.cr_lb: float = 0.0


class Compressibility:
    def __init__(self, input_data: np.ndarray, bins: int = 1024):
        self.data = input_data
        self.bins = bins
        _tmp = np.histogram(input_data, np.arange(0, bins))
        self.hist = _tmp[0]
        self.entropy = 0.0
        self.sum = 0
        self.huff_est = HuffEstStruct()
        self.most_likely_prob = 0.0

    def get_entropy_and_huffman_est(self):
        self.sum = np.sum(self.hist)
        prob = self.hist / self.sum
        entropy = reduce(
            lambda e1, e2: e1 + e2,
            list(map(
                lambda p: -p * np.log2(p) if p != 0.0 else 0,
                prob
            )))
        self.huff_est.entropy = entropy
        most_likely_freq = np.max(self.hist)
        most_likely_prob = most_likely_freq / self.sum
        self.most_likely_prob = most_likely_prob

        self.huff_est.redundancy_lb = 1 - reduce(
            lambda e1, e2: e1 + e2,
            list(map(
                lambda p: -p * np.log(p),
                [most_likely_prob, 1 - most_likely_prob]
            )))
        self.huff_est.avg_bitlen_lb = entropy + self.huff_est.redundancy_lb
        self.huff_est.cr_ub = 32 / self.huff_est.avg_bitlen_lb

        self.huff_est.redundancy_ub = most_likely_prob + 0.086
        self.huff_est.avg_bitlen_ub = entropy + self.huff_est.redundancy_ub
        self.huff_est.cr_lb = 32 / self.huff_est.avg_bitlen_ub


class AnalyzeSmoothnessWholeDatum:

    def __init__(self,
                 input_data: np.ndarray,
                 dataset_constant: datasets.Constant,
                 dataset_block: datasets.Block,
                 nsample: int = 1000,
                 max_dist: int = 200,
                 hist_bins: int = 1024,
                 estimate_c13y: bool = False
                 ):
        """Analyze over the whole datum.

        Args:
            dataset_constant (datasets.Constant): dataset metadata
            dataset_block (datasets.Block): chunking information
            nsample (int, optional): number of sample. Defaults to 1000.
            max_dist (int, optional): maximum distance. Defaults to 200.
        """
        self.data = input_data
        self.nsample = nsample
        self.max_dist = max_dist
        self.constant = dataset_constant
        self.block = dataset_block
        self.var_binary = None
        self.var_abs_1d = None
        self.var_pwr_1d = None
        self.var_abs_2d = None
        self.var_pwr_2d = None
        self.count = np.zeros(self.max_dist + 1, dtype=np.int32)

        self.c13y = None

        if estimate_c13y:
            self.c13y = Compressibility(self.data, hist_bins)
            self.c13y.get_entropy_and_huffman_est()

        self.sample = []

    def run__random_start__random_dist_1d(self, mode="binary_variance", pwr=False):
        """random start point in the range of [0, data-len] (1D index), random distance in the range of [1, predefined-max-dist]

        Args:
            mode (str, optional): variance in mean square or mean binary, defaults to "binary_variance"
            pwr: if calculating pwr
        """
        data = self.data.copy()
        data = data.flatten()
        # sample pairs
        list_start = np.random.randint(0, self.constant.size - 1, self.nsample)
        list_dist = np.random.randint(0, self.max_dist, self.nsample)
        self.sample = [(s, d) for (s, d) in zip(list_start, list_dist)]
        # process
        if mode == "binary_variance":
            self.var_binary = np.zeros(self.max_dist + 1, dtype=np.float32)
            for (s, d) in self.sample:
                if s + d >= self.constant.size:
                    continue
                self.count[d] += 1
                res = int(data[s + d] != data[s])
                self.var_binary[d] += res
            for i, c in enumerate(self.count):
                if c != 0:
                    self.var_binary[i] /= c
        elif mode == "conventional_variance":
            self.var_abs_1d = np.zeros(self.max_dist + 1, dtype=np.float32)
            if pwr:
                self.var_pwr_1d = np.zeros(self.max_dist + 1, dtype=np.float32)
            for (s, d) in self.sample:
                if s + d >= self.constant.size:
                    continue
                self.count[d] += 1
                res = int(data[s + d]) - int(data[s])
                self.var_abs_1d[d] += abs(res)
                if pwr:
                    self.var_pwr_1d[d] += res ** 2
            for i, c in enumerate(self.count):
                if c != 0:
                    self.var_abs_1d[i] /= c
                    if pwr:
                        self.var_pwr_1d[i] /= c

    def run__random_start__random_dist_2d(self, mode="binary_variance", pwr=False):
        """random start point in the range of [0, data-len] (1D index), random distance in the range of [1, predefined-max-dist]

        Args:
            mode (str, optional): variance in mean square or mean binary, defaults to "binary_variance"
            pwr: if calculating pwr
        """
        data = self.data.copy()

        # rectangle-(quarter-incircle) overhead:
        # (r ** 2) / (1/4 * np.pi * (r ** 2)) = 4 / np.pi
        nsample_extra = int(4 / np.pi * self.nsample)
        max_dist_squared = self.max_dist ** 2

        # sample pairs
        list_start_x = np.random.randint(0, self.constant.x - 1, nsample_extra)
        list_start_y = np.random.randint(0, self.constant.y - 1, nsample_extra)
        list_delta_x = np.random.randint(0, self.max_dist, nsample_extra)
        list_delta_y = np.random.randint(0, self.max_dist, nsample_extra)
        self.sample = [((y, x), (dy, dx)) for ((y, x), (dy, dx)) in
                       zip(
                           zip(list_start_y, list_start_x),
                           zip(list_delta_y, list_delta_x)
                       )
                       if dy ** 2 + dx ** 2 <= max_dist_squared
                       and ((y + dy) < self.constant.y and (x + dx) < self.constant.x)
                       ]
        # process
        if mode == "conventional_variance":
            self.var_abs_2d = np.zeros(self.max_dist + 1, dtype=np.float32)
            if pwr:
                self.var_pwr_2d = np.zeros(self.max_dist + 1, dtype=np.float32)
            for (y, x), (dy, dx) in self.sample:
                dist = int(np.sqrt(dy ** 2 + dx ** 2))  # floor
                self.count[dist] += 1
                res = int(data[y + dy, x + dx]) - int(data[y, x])
                self.var_abs_2d[dist] += abs(res)
                if pwr:
                    self.var_pwr_2d[dist] += res ** 2
            for i, c in enumerate(self.count):
                if c != 0:
                    self.var_abs_2d[i] /= c
                    if pwr:
                        self.var_pwr_2d[i] /= c

    def run__random_start__enumerate_dist_1d(self):
        """random start point in the range of [0, data-len] (1D index), distance interated 1:predefined-max-dist

        Note:
            result not stable, use with caution
        """

        # sample pairs
        self.sample = np.random.randint(0, self.constant.size - 1, self.nsample)
        # process
        data = self.data.copy()
        data = data.flatten()

        for idx in self.sample:
            start = data[idx]
            for dist in range(1, self.max_dist):
                if idx + dist >= self.constant.size:
                    break
                res = int(data[idx + dist] != start)
                self.var_binary[dist] += res
        self.var_binary /= self.nsample
        self.var_abs_1d = np.sqrt(self.var_binary)


class AnalyzeSmoothnessBlockwise:
    def __init__(self,
                 dataset_constant: datasets.Constant,
                 dataset_block: datasets.Block,
                 nsample: int = 20):
        """[summary]

        Args:
            dataset_constant (datasets.Constant): [description]
            dataset_block (datasets.Block): [description]
            nsample (int, optional): [description]. Defaults to 20.
        """
        self.nsample = nsample
        self.constant = dataset_constant
        self.block = dataset_block
        self.var_abs = None
        self.var_pwr = None

    def sample_datablock(
            self) -> List[Union[int, Tuple[int, int], Tuple[int, int, int]]]:
        list_tuple = None
        if self.constant.ndim == 1:
            list_x = np.random.randint(0, self.block.nx - 1, self.nsample)
            list_tuple = list(list_x)
        elif self.constant.ndim == 2:
            list_x = np.random.randint(0, self.block.nx - 1, self.nsample)
            list_y = np.random.randint(0, self.block.ny - 1, self.nsample)
            list_tuple = list(zip(list_y, list_x))
        elif self.constant.ndim == 3:
            list_x = np.random.randint(0, self.block.nx - 1, self.nsample)
            list_y = np.random.randint(0, self.block.ny - 1, self.nsample)
            list_z = np.random.randint(0, self.block.nz - 1,
                                       self.nsample)  # exclude the boundary
            list_tuple = list(zip(list_z, list_y, list_x))
        else:
            raise ValueError("ndim must be in {1,2,3}")

        list_tuple = sorted(set(list_tuple))
        print(
            f"sampling fraction: {self.nsample / self.block.total_num * 100:.2f}%"
        )
        return list_tuple

    def localize_block(
            self, data: np.ndarray, block_id: Union[int, Tuple[int, int],
                                                    Tuple[int, int,
                                                          int]]) -> np.ndarray:
        res = np.zeros(self.block.length, dtype=np.float32)
        """[summary]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """

        if self.constant.ndim == 1:
            x = block_id * self.block.side
            res = data[x:x + self.block.side]
        elif self.constant.ndim == 2:
            res = res.reshape((self.block.side, self.block.side))
            y = block_id[0] * self.block.side
            x = block_id[1] * self.block.side
            for iy in range(self.block.side):
                target = iy * self.block.stride1
                res[iy, :] = data[y + iy, x:x + self.block.side]
        elif self.constant.ndim == 3:
            res = res.reshape(
                (self.block.side, self.block.side, self.block.side))
            z = block_id[0] * self.block.side
            y = block_id[1] * self.block.side
            x = block_id[2] * self.block.side
            for iz in range(self.block.side):
                for iy in range(self.block.side):
                    res[iz, iy, :] = data[z + iz, y + iy,
                                     x:x + self.block.side]
        else:
            raise ValueError("ndim must be in {1,2,3}")
        return res

    # TODO range
    def impl_get_variance_linearized1d(self,
                                       chunk: np.ndarray,
                                       mode: str = "abs"):
        """[summary]

        Args:
            chunk (np.ndarray): [description]
            mode (str, optional): [description]. Defaults to "abs".

        Returns:
            [type]: [description]
        """
        length = self.block.length
        variance = np.zeros(self.block.length - 1)

        if self.constant.ndim != 1:
            buffer = chunk.flatten()

        if mode == "abs":
            for i, val in enumerate(buffer):
                if i == 0:
                    continue  # skip the extruding first point
                for j in range(1, length - i - 1):  # max: l-1
                    variance[j] += abs(buffer[i + j] - buffer[i])
        elif mode == "pwr":
            for i, val in enumerate(buffer):
                if i == 0:
                    continue  # skip the extruding first point
                for j in range(1, length - i - 1):  # max: l-1
                    variance[j] += (buffer[i + j] - buffer[i]) ** 2
        return variance

    @staticmethod
    def helper_get_diag(a, b) -> int:
        diag = np.ceil(np.sqrt(a ** 2 + b ** 2))
        diag = int(diag)
        return diag

    # TODO range
    def impl_get_variance_2d(self, chunk: np.ndarray, mode: str = "abs"):
        """[summary]

        Args:
            chunk (np.ndarray): [description]
            mode (str, optional): [description]. Defaults to "abs".

        Returns:
            [type]: [description]
        """

        buffer = chunk.copy()

        len_diag = self.helper_get_diag(self.block.side, self.block.side)
        variance_dist = np.zeros(len_diag - 1)
        count_dist = np.zeros(len_diag - 1)

        if mode == "abs":
            for iy in range(self.block.side):
                for ix in range(self.block.side):
                    if iy == 0 or ix == 0:
                        continue
                    for iiy in range(0, self.block.side - iy - 1):
                        for iix in range(0, self.block.side - ix - 1):
                            if iiy == 0 and iix == 0:
                                continue
                            rounded_bin = self.helper_get_diag(iiy, iix)
                            variance_dist[rounded_bin] += abs(
                                buffer[iy + iiy, ix + iix] - buffer[iy, ix])
                            count_dist[rounded_bin] += 1
        elif mode == "pwr":
            for iy in range(self.block.side):
                for ix in range(self.block.side):
                    if iy == 0 and ix == 0:
                        continue
                    for iiy in range(0, self.block.side - iy - 1):
                        for iix in range(0, self.block.side - ix - 1):
                            if iiy == 0 and iix == 0:
                                continue
                            rounded_bin = self.helper_get_diag(iiy, iix)
                            variance_dist[rounded_bin] += (
                                                                  buffer[iy + iiy, ix + iix] - buffer[iy, ix]) ** 2
                            count_dist[rounded_bin] += 1
        return variance_dist, count_dist

    def run(self, input_data: np.ndarray, mode: str = "1d"):
        """[summary]

        Args:
            input_data (np.ndarray): [description]
            mode (str, optional): [description]. Defaults to "1d".
        """
        choices = self.sample_datablock()

        var_abs, var_pwr = None, None

        # averaging
        if mode == "1d":
            length = self.block.length
            var_abs = np.zeros(length - 1)
            var_pwr = np.zeros(length - 1)

            for block_id in choices:
                one_block = self.localize_block(input_data, block_id)
                var_abs += self.impl_get_variance_linearized1d(one_block)
                var_pwr += self.impl_get_variance_linearized1d(
                    one_block, "pwr")
            for i, (_abs, _pwr) in enumerate(zip(var_abs, var_pwr)):
                count = (length - i) * len(choices)
                var_abs[i] /= count
                var_pwr[i] /= count
        elif mode == "2d":  # not for encoding
            len_diag = self.helper_get_diag(self.block.side, self.block.side)
            var_abs = np.zeros(len_diag - 1)
            var_pwr = np.zeros(len_diag - 1)
            count_2d = np.zeros(len_diag - 1)

            for block_id in choices:
                one_block = self.localize_block(input_data, block_id)
                res_var_abs, res_count = self.impl_get_variance_2d(one_block)
                res_var_pwr, _ = self.impl_get_variance_2d(one_block, "pwr")
                var_abs += res_var_abs
                var_pwr += res_var_pwr
                count_2d += res_count

                # print(res_count)
            for i, (c, _abs, _pwr) in enumerate(zip(count_2d, var_abs,
                                                    var_pwr)):
                # print(c)
                if c == 0:
                    continue
                var_abs[i] /= c
                var_pwr[i] /= c

        self.var_abs = var_abs
        self.var_pwr = var_pwr

    def print(self):
        print(self.block.__dict__)
