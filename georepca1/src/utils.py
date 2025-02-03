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


def load_dat(animal, p, to_convert=["envs", "position", "trace"], format="MATLAB"):
    """
    Load dataset from target animal (matlab file), convert environment labels, position, and trace data
    to numpy arrays. Can load original MATLAB files, or saved joblib file with converted fields
    :param animal: animal ID (e.g., "QLAK-CA1-08")
    :param p: path of parent folder as string (e.g., "User/yourname/Desktop/georepca1")
    :param to_convert: fields to convert to numpy arrays
    :param format: format of data file to be loaded, either "MATLAB" or "joblib"
    :return: dataset in nested dictionary under animal key
    """
    # Define path to datasets (should be added to "data" folder after downloading)
    p_data = os.path.join(p, "data")
    print(f'Loading preprocessed data for animal {animal}')
    # if file format is .mat (MATLAB), then load with mat73.loadmat and convert necessary fields to numpy arrays
    if format == "MATLAB":
        dataset = loadmat(os.path.join(p_data, f"{animal}.mat"))
        # convert position, trace, and envs data from lists to numpy matrices
        for key in to_convert:
            dataset[key] = np.array(dataset[key])
        return {animal: dataset}
    # if file format is instead joblib, load the pre-saved joblib file version (previously with save_dat)
    elif format == "joblib":
        dataset = joblib.load(os.path.join(p_data, f"{animal}"))
        return dataset


def save_dat(dat, animal, p):
    """
    :param dat: 
    :param animal: 
    :param p: 
    :return: 
    """
    p_data = os.path.join(p, "data")
    joblib.dump(dat, os.path.join(p_data, animal))


def mat2joblib(animal, p):
    """
    Load original MATLAB file and save joblib file with converted fields
    """
    dat = load_dat(animal, p)
    save_dat(dat, animal, p)


def trace_sfps(SFPs):
    SFPs = deepcopy(SFPs)
    n_cells, n_days = SFPs.shape[2], SFPs.shape[3]
    SFPs_traced = np.zeros_like(SFPs)
    for c in range(n_cells):
        for d in range(n_days):
            SFPs_bin = SFPs[:, :, c, d] > 1e-3
            SFPs_traced[:, :, c, d] = (np.vstack((np.zeros(SFPs_bin.shape[0])[np.newaxis], np.diff(SFPs_bin, axis=0))) +
                                       np.hstack((np.zeros(SFPs_bin.shape[1])[np.newaxis].T,
                                                  np.diff(SFPs_bin, axis=1)))).astype(bool)
    return SFPs_traced.astype(int)