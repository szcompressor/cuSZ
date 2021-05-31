__author__ = "Jiannan Tian"
__copyright__ = "(C) 2021 by Washington State University, Argonne National Laboratory"
__license__ = "BSD 3-Clause"
__version__ = "0.2.x"
__date__ = "2021-03-30"

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# plt.style.use('ggplot')
mpl.rcParams['figure.dpi'] = 300

from smoothness import *
from vis import *

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:


eb_str = "1e-4"
eb = 1e-2

quant_folder = os.path.join(os.getenv("HOME"), "sdrb-cesm", "single")
data_folder = os.path.join(os.getenv("HOME"), "sdrb-cesm", "single")
cesm_constant = datasets.Constant(**datasets.cesm)
cesm_block = datasets.Block(cesm_constant)

quant_fields  = [os.path.join(quant_folder, f) for f in os.listdir(quant_folder) if f.endswith("quant")]
origin_fields = [os.path.join(data_folder, f)  for f in os.listdir(data_folder)  if f.endswith("dat")  ]

quant_fields = sorted(quant_fields)
origin_fields = sorted(origin_fields)


# debug
# print("\n".join(quant_fields))
# print("\n".join(origin_fields))

# debug: stop here
# sys.exit()

for _i, (name_ori, name_qua) in enumerate(list(zip(origin_fields, quant_fields))):
    name = name_ori.split("/")[-1].split(".")[0]
    print(f"{_i}\tprocessing {name}".ljust(24), end=",\t")

    data_ori = np.fromfile(name_ori, dtype=np.float32).reshape(cesm_constant.y, cesm_constant.x)
    rng = np.max(data_ori) - np.min(data_ori)
    ebx2 = eb * rng * 2
    data_ori /= ebx2
    data_qua = np.fromfile(name_qua, dtype=np.uint16).reshape(cesm_constant.y, cesm_constant.x)

    # original data
    ana_ori = AnalyzeSmoothnessWholeDatum(data_ori, cesm_constant, cesm_block, 10000)
    ana_ori.run__random_start__random_dist_2d("conventional_variance")
    ana_ori.run__random_start__random_dist_1d("conventional_variance")
    # quant code
    ana_qua = AnalyzeSmoothnessWholeDatum(data_qua, cesm_constant, cesm_block, 10000, estimate_c13y=True)
    ana_qua.run__random_start__random_dist_2d("conventional_variance")
    ana_qua.run__random_start__random_dist_1d("conventional_variance")
    ana_qua.run__random_start__random_dist_1d("binary_variance")

    max_dist = ana_ori.max_dist

    fig_output_path = os.path.join(f"{data_folder}")
    # debug
    # print(os.path.join(fig_output_path, name))

    process_plot_new3fig(ana_ori, ana_qua, max_dist, fig_output_path, name)
    # break  # test the first iteration
print()
