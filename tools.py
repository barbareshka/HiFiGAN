import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import torch
import shutil

def scan(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]
