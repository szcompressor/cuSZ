from _byte_shfl import *
from os.path import expanduser
import os

home = expanduser("~")

def run_exp_byte_shfl(dataset, zyx):
    joined_data_path = os.path.join(home, "data", dataset)
    datafields = [i for i in os.listdir(joined_data_path) if i.endswith(".f32")]
    datafields = sorted(datafields)
    for d in datafields:
        byte_shuffle(os.path.join(joined_data_path, d), zyx)


run_exp_byte_shfl("cesm_dim3-3600x1800", (1800, 3600))
# run_exp_byte_shfl("aramco-data", (449, 449, 235))
# run_exp_byte_shfl("280953867", (280953867,))
# run_exp_byte_shfl("nyx-512x512x512", (512, 512, 512))
# run_exp_byte_shfl("xyz_500x500x100_hurricane", (100, 500, 500))
