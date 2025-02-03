import os
import sys
import smtpd
import warnings
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings('ignore')
import scipy.stats.morestats
import matplotlib
from scipy.ndimage import gaussian_filter
import seaborn as sns
sns.set(style='dark', font_scale=1.25)

from copy import deepcopy
from glob import glob
import scipy
from mat73 import loadmat
import joblib
from tqdm import tqdm
import numpy as np


def load_dat(animal, p, to_convert=["envs", "position", "trace"]):
    """
    Load dataset from target animal (matlab file), convert environment labels, position, and trace data
    to numpy arrays.
    :param animal: animal ID (e.g., "QLAK-CA1-08)
    :param p: path of parent folder as string (e.g., "User/yourname/Desktop/georepca1")
    :param to_convert:
    """
    p_data = os.path.join(p, "data")
    print(f'Loading preprocessed data for animal {animal}')
    dataset = loadmat(os.path.join(p_data, f"{animal}.mat"))
    # convert position, trace, and envs data from lists to numpy matrices
    for key in to_convert:
        dataset[key] = np.array(dataset[key])
    return {animal: dataset}
