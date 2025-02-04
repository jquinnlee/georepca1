import os
import sys
import smtpd
import warnings
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings('ignore')
import scipy.stats.morestats
import matplotlib
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import MDS
import torch
from torch.nn import AvgPool1d
import seaborn as sns
sns.set(style='dark', font_scale=1.25)
from scipy.stats import *
from copy import deepcopy
from glob import glob
import scipy
from mat73 import loadmat
import joblib
from tqdm import tqdm
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.proportion import proportions_chisquare as chisquare
from scipy.stats import ttest_1samp
from scipy.stats import sem
import matplotlib.path as mpath
import matplotlib.patches as patches
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.base import BaseEstimator
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state, check_array, check_symmetric
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import _deprecate_positional_args
from joblib import Parallel, delayed, effective_n_jobs


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
    """
    Create contours from centred spatial footprints ("SFPs") previously registered across days with CellReg
    :param SFPs: x- and y-centred SFPs
    :return: SFP contours
    """
    SFPs = deepcopy(SFPs)
    n_cells, n_days = SFPs.shape[2], SFPs.shape[3]
    SFPs_traced = np.zeros_like(SFPs)
    for c in range(n_cells):
        for d in range(n_days):
            # create contours of SFPs by detecting changes in values at edge of solid object in SFPs matrices
            SFPs_bin = SFPs[:, :, c, d] > 1e-3
            SFPs_traced[:, :, c, d] = (np.vstack((np.zeros(SFPs_bin.shape[0])[np.newaxis], np.diff(SFPs_bin, axis=0))) +
                                       np.hstack((np.zeros(SFPs_bin.shape[1])[np.newaxis].T,
                                                  np.diff(SFPs_bin, axis=1)))).astype(bool)
    return SFPs_traced.astype(int)


def get_environment_label(env_name, flipud=False):
    '''
    get_environment label will return a custom maker from environment name as input that can be used for plotting
    '''
    codes = False
    # First draw vertices that will create the shape of the environment based on input name (env names in weirdGeos dat)
    if env_name == 'square':
        polys = np.array([[0, 0], [0, 30], [30, 30], [30, 0], [0, 0]])-15
    elif env_name == 'o':
        polys_outside = np.array([[0, 0], [0, 30], [30, 30], [30, 0], [0, 0]]) - 15
        polys_inside = np.array([[10, 10], [10, 21], [21, 21], [21, 10], [10, 10]]) - 15
        polys = np.concatenate((polys_outside[::1], polys_inside[::-1]))
        codes = np.ones(polys.shape[0], dtype=mpath.Path.code_type) * mpath.Path.LINETO
        codes[0] = mpath.Path.MOVETO
        codes[5] = mpath.Path.MOVETO
    elif env_name == 't':
        polys = np.array([[10, 0], [10, 21], [0, 21], [0, 30], [30, 30], [30, 21], [21, 21], [21, 0], [10, 0]]) - 15
    elif env_name == 'u':
        if flipud:
            polys = np.array([[0, 0], [0, 10], [21, 10], [21, 21], [0, 21], [0, 30], [30, 30], [30, 0], [0, 0]]) - 15
        else:
            polys = np.array([[0, 0], [30, 0], [30, 10], [10, 10], [10, 21], [30, 21], [30, 30], [0, 30], [0, 0]]) - 15
    elif env_name == 'rectangle':
        polys = np.array([[10, 0], [30, 0], [30, 30], [10, 30], [10, 0]]) - 15
    elif env_name == '+':
        polys = np.array([[0, 10], [10, 10], [10, 0], [21, 0], [21, 10], [30, 10], [30, 21], [21, 21], [21, 30],
                          [10, 30], [10, 21], [0, 21], [0, 10]]) - 15
    elif env_name == 'i':
        polys = np.array([[0, 0], [30, 0], [30, 10], [21, 10], [21, 21], [30, 21], [30, 30], [0, 30], [0, 21],
                          [10, 21], [10, 10], [0, 10], [0, 0]]) - 15
    elif env_name == 'l':
        if flipud:
            polys = np.array([[0, 0], [0, 10], [21, 10], [21, 30], [30, 30], [30, 0], [0, 0]]) - 15
        else:
            polys = np.array([[0, 0], [0, 30], [10, 30], [10, 10], [30, 10], [30, 0], [0, 0]]) - 15
    elif env_name == 'bit donut':
        if flipud:
            polys = np.array([[0, 0], [30, 0], [30, 21], [21, 21], [21, 10], [10, 10], [10, 21], [21, 21], [21, 30],
                              [0, 30], [0, 0]]) - 15
        else:
            polys = np.array([[0, 0], [30, 0], [30, 30], [10, 30], [10, 21], [21, 21], [21, 10], [10, 10], [10, 21],
                                  [0, 21], [0, 0]]) - 15
    elif env_name == 'glenn':
        if flipud:
            polys = np.array([[0,  0], [0, 21], [9, 21], [9, 30], [30, 30], [30, 9], [21, 9], [21, 0], [0, 0]]) - 15
        else:
            polys = np.array([[10, 0], [30, 0], [30, 21], [21, 21], [21, 30], [0, 30], [0, 10], [10, 10], [10, 0]]) - 15
    # Use Path function to create image path that matplotlib can plot
    if np.any(codes):
        if flipud:
            return mpath.Path(np.fliplr(polys), codes), -1 * polys
        else:
            return mpath.Path(np.fliplr(polys), codes), polys
    else:
        if flipud:
            return mpath.Path(np.fliplr(polys)), -1 * polys
        else:
            return mpath.Path(np.fliplr(polys)), polys


########################################################################################################################
# Rate map generation and split-half reliability measures
def get_rate_maps(position, trace, n_bins=15, fps=30, buffer=1e-5, filter_size=1.5):
    # time is row axis for position and trace data
    # First determine the length of the recording and number of neurons
    len_recording = trace.shape[0]
    n_cells = trace.shape[1]

    # Initialize rate maps and count map (which will count the number of times each bin is visited)
    rate_maps = np.zeros([n_cells, n_bins, n_bins])
    count_map = np.zeros([n_bins, n_bins])

    # Create a binned version of the position data with integer division
    position_binned = (position // ((np.nanmax(position, axis=0) + buffer) / n_bins)).astype(int)

    # Iterate through binned position data and add value of each trace
    # to rate map and a one to the same position on the count map
    for t, (x, y) in enumerate(position_binned):
        rate_maps[:, x, y] += trace[t, :]
        count_map[x, y] += 1

    nan_map = np.zeros_like(count_map) * np.nan
    nan_map[np.where(count_map)[0], np.where(count_map)[1]] = 1.

    if filter_size:
        rate_maps = gaussian_filter1d(gaussian_filter1d(rate_maps, sigma=filter_size, axis=1),
                                      sigma=filter_size, axis=2)
        count_map = gaussian_filter1d(gaussian_filter1d(count_map, sigma=filter_size, axis=0),
                                      sigma=filter_size, axis=1)

    # To determine the firing rate of neurons in each bin, divide by the number of
    # times the animal was in that location and multiply by the frames per second
    for cell in range(n_cells):
        rate_maps[cell] = (rate_maps[cell] / count_map) * fps * nan_map

    # Also create a probability map of dwell times in each bin to return
    occupancy_map = count_map / len_recording

    average_firing = (np.sum(trace, axis=0) / trace.shape[0]) * fps

    return rate_maps, occupancy_map, average_firing


def get_split_half(position, trace):

    # get the total length of the recording and number of cells
    len_recording = trace.shape[0]
    n_cells = trace.shape[1]

    # define first and second half session range in time series
    h1 = np.arange(len_recording / 2).astype(int)
    h2 = np.arange(len_recording / 2, len_recording).astype(int)

    # build rate maps for first and second session halves
    maps_h1, _, _ = get_rate_maps(position[h1], trace[h1])
    maps_h2, _, _ = get_rate_maps(position[h2], trace[h2])

    # get the number of bins in both x and y dimensions
    n_bins_x, n_bins_y = np.maximum(maps_h1.shape[1], maps_h2.shape[1]),\
                         np.maximum(maps_h1.shape[2], maps_h2.shape[2])

    # linearize rate maps for both session halves and assign them to first and second idx in last axis
    flat_maps = np.zeros([n_bins_x * n_bins_y, n_cells, 2])
    for cell in range(n_cells):
        flat_maps[:, cell, 0] = maps_h1[cell, :, :].flatten()
        flat_maps[:, cell, 1] = maps_h2[cell, :, :].flatten()

    # exclude spatial bins that were not visited in both session halves
    flat_maps[np.isnan(np.sum(flat_maps, axis=2)), :] = np.nan

    # correlate first and second session halves leaving out nan values
    map_corr = np.zeros(n_cells)

    for cell in range(n_cells):
        if np.any(~np.isnan(flat_maps[:, cell, :])):
            map_corr[cell], _ = pearsonr(x=flat_maps[~np.isnan(flat_maps[:, cell, 0]), cell, 0],
                                         y=flat_maps[~np.isnan(flat_maps[:, cell, 1]), cell, 1])
        else:
            map_corr[cell] = np.nan

    return map_corr


def get_shuffle_split_half(position, trace, nsims=1000, min_time=30, fps=30):

    # get the length of the recording and number of cells
    len_recording = trace.shape[0]
    n_cells = trace.shape[1]

    # initialize shuffle_corr
    shuffle_corr = np.zeros([n_cells, nsims])

    # circularly shuffle (roll) behavioural data randomly > min_time away from true time, and compute split-half corr
    for i in tqdm(range(nsims), leave=True, position=0, desc='Computing split-half reliability on shuffled data'):
        shuffled_position = np.roll(position, np.random.randint(min_time * fps, len_recording - min_time * fps), axis=0)
        shuffle_corr[:, i] = get_split_half(shuffled_position, trace)

    return shuffle_corr


def get_place_cells(position, trace, nsims=500, alpha=0.05):

    n_cells = trace.shape[1]
    sh_actual = get_split_half(position, trace)
    sh_shuffle = get_shuffle_split_half(position, trace, nsims)
    p_vals = np.zeros(n_cells)
    place_cells = np.zeros(n_cells).astype(bool)
    for cell in range(n_cells):
        p_vals[cell] = 1 - ((sh_actual[cell] > sh_shuffle[cell, :]).sum() / nsims)
        if p_vals[cell] < alpha:
            if ~np.isnan(p_vals[cell]):
                place_cells[cell] = True
    p_vals[np.isnan(trace[0, :])] = np.nan
    place_cells[np.isnan(trace[0, :])] = np.nan

    return p_vals, place_cells


def get_shr_within(dat, animal, nsims=1000):
    n_days = dat[animal]['trace'].shape[0]
    place_cells = [None] * n_days
    p_vals = [None] * n_days
    for day in range(n_days):
        p_vals[day], place_cells[day] = get_place_cells(dat[animal]['position'][day].T,
                                                        dat[animal]['trace'][day].T, nsims=nsims)
    p_vals, place_cells = np.array(p_vals).T, np.array(place_cells).T
    return p_vals, place_cells


########################################################################################################################
# Representational similarity functions for RSM construction
def get_cell_rsm(maps, down_sample_mask=None, unsmoothed=False, d_thresh=0):
    # first load in either smoothed or unsmoothed rate maps depending on input arg
    if unsmoothed:
        # when loading in transpose the maps such that the cell and day axes come first
        maps = deepcopy(maps['unsmoothed']).transpose((2, 3, 0, 1))
    else:
        maps = deepcopy(maps['smoothed']).transpose((2, 3, 0, 1))
    # then use a down-sampling mask to nan out cells if down_sample_mask is
    # provided with a masking (n_cells, n_days)
    if np.any(down_sample_mask):
        maps[~down_sample_mask] = np.nan
    # transpose the rate maps back to their original order (xdim, ydim, ncell, ndays)
    maps = maps.transpose((2, 3, 0, 1))
    n_days = maps.shape[-1]
    # flatten the rate maps
    flat_maps = maps.reshape(maps.shape[0] * maps.shape[1], maps.shape[2], maps.shape[3])
    # create a mask to avoid correlating nans
    nan_idx = np.isnan(flat_maps)
    reg_idx = np.any(~nan_idx, axis=0)
    cell_idx = np.where(reg_idx.sum(axis=1) >= d_thresh)[0]
    cell_rsm = np.zeros([n_days, n_days, cell_idx.shape[0]])
    for c, cell in tqdm(enumerate(cell_idx), leave=True, position=0,
                        desc='Correlating ratemaps across days and cell pairs'):
        for s1 in range(n_days):
            for s2 in range(n_days):
                s1_map, s2_map = flat_maps[:, cell, s1], flat_maps[:, cell, s2]
                if np.any(~np.isnan(s1_map)) and np.any(~np.isnan(s2_map)):
                    nan_mask = ~np.isnan(np.vstack((s1_map, s2_map)).sum(axis=0))
                    cell_rsm[s1, s2, c] = pearsonr(s1_map[nan_mask],
                                                   s2_map[nan_mask])[0]
                else:
                    cell_rsm[s1, s2, c] = np.nan
    return cell_rsm, cell_idx


def get_mean_map_corr(animals, p):
    p_data = os.path.join(p, "data")
    # build rate map correlations across geometries, averaging first within animals across sequences, then across animal
    map_corr_animals_sequences = np.zeros([len(animals), 3, 11, 11]) * np.nan
    for a, animal in enumerate(animals):
        # temp = joblib.load(os.path.join(p_data, f"{animal}_rate_maps"))
        temp = load_dat(animal, p, format="joblib")
        maps, envs = temp[animal]["maps"], temp[animal]["envs"].ravel()
        s_days = np.vstack((np.where(envs == "square")[0][:-1], np.where(envs == "square")[0][1:])).T
        if a == 0:
            cannon_order = envs[1:10]
            labels = np.array(["square"] + list(cannon_order) + ["square"])
        else:
            for s1, s2 in s_days:
                reorder_idx = np.array([np.where(envs[s1+1:s2] == e)[0]+(s1+1) for e in cannon_order]).ravel()
                maps["smoothed"] = maps["smoothed"].transpose()
                maps["smoothed"][s1+1:s2] = maps["smoothed"][reorder_idx]
                maps["smoothed"] = maps["smoothed"].T
        # build rsm
        map_corr, _ = get_cell_rsm(maps)
        for seq, (s1, s2) in enumerate(s_days):
            map_corr_animals_sequences[a, seq, :, :] = np.nanmean(map_corr[s1:s2 + 1, s1:s2 + 1], axis=-1)
            if seq == 0:
                seq_mean = np.nanmean(map_corr[s1:s2+1, s1:s2+1], axis=-1)[:, :, np.newaxis]
            else:
                seq_mean = np.dstack((seq_mean, np.nanmean(map_corr[s1:s2+1, s1:s2+1], axis=-1)[:, :, np.newaxis]))

        if a == 0:
            animals_mean = np.nanmean(seq_mean, axis=-1)[:, :, np.newaxis]
        else:
            animals_mean = np.dstack((animals_mean, np.nanmean(seq_mean, axis=-1)[:, :, np.newaxis]))
    mean_map_corr = animals_mean.mean(-1)
    return mean_map_corr, cannon_order, labels, map_corr_animals_sequences


def get_rsm_similarity_animals_sequences(animals, p, nsims=100, s_prop=0.9):
    np.random.seed(2023)
    map_corr_envs = joblib.load(os.path.join(p, "results", "map_corr_envs"))
    map_corr_animals_sequences = map_corr_envs["map_corr_animals_sequences"]
    cols = ["Animal A", "Animal B", "Sequence", "Fit", "Shuffle"]
    map_corr_animal_sequence_similarity = np.zeros([len(animals) ** 2 * 3 * nsims * 2, len(cols)]) * np.nan
    c = 0
    for s in range(map_corr_animals_sequences.shape[1]):
        for a1 in tqdm(range(len(animals)), desc="Calculating rate map correlation rsm similarity across animals",
                       position=0, leave=True):
            for a2 in range(len(animals)):
                temp_corr1, temp_corr2 = map_corr_animals_sequences[a1, s, :, :], map_corr_animals_sequences[a2, s, :,
                                                                                  :]
                for i in range(nsims):
                    choice_idx = np.argwhere(np.tri(temp_corr1.shape[0], k=-1))
                    choice_idx = choice_idx[np.random.choice(np.arange(choice_idx.shape[0]),
                                                             size=int(choice_idx.shape[0] * s_prop),
                                                             replace=True)]
                    if np.any(~np.isnan(temp_corr1)) and np.any(~np.isnan(temp_corr2)):
                        actual = kendalltau(temp_corr1[choice_idx], temp_corr2[choice_idx])[0]
                        shuffle = kendalltau(np.random.permutation(temp_corr1)[choice_idx], temp_corr2[choice_idx])[0]
                        map_corr_animal_sequence_similarity[c] = np.hstack((a1, a2, s + 1, actual, False))
                        c += 1
                        map_corr_animal_sequence_similarity[c] = np.hstack((a1, a2, s + 1, shuffle, True))
                        c += 1

    df_map_corr_animals_sequences = pd.DataFrame(data=map_corr_animal_sequence_similarity, columns=cols)
    df_map_corr_animals_sequences.dropna(axis=0, inplace=True)
    df_map_corr_animals_sequences = df_map_corr_animals_sequences[df_map_corr_animals_sequences["Animal A"] !=
                                                                  df_map_corr_animals_sequences["Animal B"]]
    return df_map_corr_animals_sequences


########################################################################################################################
# Bayesian decoding of animal position within sessions
def fit_decoder(behav, traces, feature_max, temporal_bin_size=3):
    # temporally bin trace and behavioral data
    pooling = AvgPool1d(kernel_size=temporal_bin_size, stride=temporal_bin_size)
    behav, traces = pooling(torch.tensor(behav.T)).numpy().astype(int).T,\
                    pooling(torch.tensor(gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0).T)).numpy().T
    # determine number of samples and behavioral features
    n_samples, n_features = behav.shape[0], behav.shape[1]
    # transform behavioral data to one-hot embedding
    onehot_behav = np.zeros(n_samples)
    empty_map = np.zeros(feature_max)
    for i in range(n_samples):
        empty_map[behav[i][0], behav[i][1]] += 1
        onehot_behav[i] = np.where(empty_map.flatten())[0]
        empty_map *= 0
    # initialize model with flat priors
    n_classes = np.unique(onehot_behav).shape[0]
    model = GaussianNB(priors=np.ones(n_classes)/n_classes)
    model.fit(traces, onehot_behav)
    return model


def test_decoder(model, behav, traces, feature_max, bin_down, temporal_bin_size=3):
    # temporally bin behav data
    pooling = AvgPool1d(kernel_size=temporal_bin_size, stride=temporal_bin_size)
    behav, traces = pooling(torch.tensor(behav.T)).numpy().astype(int).T,\
                    pooling(torch.tensor(gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0).T)).numpy().T
    # make predictions of one-hot behavior using trained model
    predictions = model.predict(traces).astype(int)
    # also return probabilities for all classes from predictions (shape n_samples X n_classes)
    # reverse-transform predictions
    n_samples, n_features = behav.shape[0], behav.shape[1]
    predictions_transform = np.zeros([n_samples, n_features])
    empty_map = np.zeros(feature_max)
    predictions_map = np.zeros([feature_max[0], feature_max[1], feature_max[0], feature_max[1]])
    for i in range(n_samples):
        trans = empty_map.flatten()
        trans[predictions[i]] += 1
        empty_map[np.where(trans.reshape(empty_map.shape))] += 1
        predictions_map[behav[i][0], behav[i][1]] += empty_map/n_samples
        predictions_transform[i] = np.array(np.where(empty_map)).squeeze()
        empty_map *= 0
    # calculate the squared distance between actual and predicted behavior
    err_squared = np.zeros(n_samples)
    for i in range(n_samples):
        err_squared[i] = (bin_down * np.linalg.norm(predictions_transform[i] - behav[i])) ** 2
    return behav, predictions_transform, err_squared, predictions_map


def decode_position_within(behav, traces, maps, n_bins=15, fps=30, v_filt_size=5,
                           v_thresh=5, cell_threshold=5, buffer=1e-15, n_fold=5):
    '''
    Provide behav and trace data with shape frames x features x days
    returns n_fold mean distance as error metric, and map of distance error for each behavioral bin
    '''

    # make a deepcopy of behav and trace data as to not change original inputs
    behav = deepcopy(behav)

    # bin down behavioural data
    behav_max = behav.max(axis=0).max(axis=1)
    bin_down = (behav_max.max() + buffer) / n_bins
    behav /= bin_down

    # determine maxima of behavioral features across all days (needed for across-day training and decoding)
    behav_max = (behav.max(axis=0).max(axis=1) + buffer).astype(int)

    # grab the number of days to iterate across, and initialize some variables to return and use as function params
    n_days = behav.shape[2]
    n_cells = traces.shape[1]

    # distances will contain the distance errors for model predictions
    distances = np.zeros([n_days, n_fold])

    # initialize velocity and cell idx for selecting data based on criteria
    cell_idx = np.zeros([n_days, n_cells]).astype(bool)
    vel_idx = np.zeros([n_days, behav.shape[0]]).astype(bool)

    # create template maps for all days to find nearest x-y bins for dist_maps after predictions are made (cleaning)
    temp_maps = np.nansum(maps, axis=2).astype(bool)

    # initialize distance maps to show distance between actual and decoded position in each bin
    mse, imse, coherence_maps = np.zeros_like(temp_maps).astype(float), np.zeros_like(temp_maps).astype(float),\
        np.zeros_like(temp_maps).astype(float)

    for d in tqdm(range(n_days), desc=f'Performing {n_fold}-fold cross-validated position decoding'):
        vel_idx[d, 1:] = gaussian_filter1d(np.linalg.norm((behav[1:, :, d] - behav[:-1, :, d]) * fps, axis=1),
                                           axis=0, sigma=v_filt_size) > (v_thresh / bin_down)
        cell_idx[d] = np.sum(traces[:, :, d][vel_idx[d]], axis=0) > cell_threshold
        target_behav = behav[:, :, d][vel_idx[d]]
        target_traces = traces[:, cell_idx[d], d][vel_idx[d]]
        kf = KFold(n_splits=n_fold)
        kf.get_n_splits(target_traces)

        for fold, (train_index, test_index) in enumerate(kf.split(target_traces)):
            # initialize maps to plot distance errors onto behavioural map
            count_map, err_squared_map, prob_map = np.zeros_like(temp_maps[:, :, d]).astype(int), \
                                                   np.zeros_like(temp_maps[:, :, d]).astype(int), \
                                                   np.zeros_like(temp_maps[:, :, d]).astype(int)
            # index target and train trace and behavioural data
            behav_train, behav_test = target_behav[train_index], target_behav[test_index]
            traces_train, traces_test = target_traces[train_index], target_traces[test_index]
            # fit the decoding model to the training data
            GNB_model = fit_decoder(behav_train, traces_train, behav_max)
            # apply the model on the test data and measure the accuracy of predictions
            # return the temporally binned behavioral data (actual), the predictions, and the distance as error metric
            actual, predictions, err_squared, predictions_dist = test_decoder(GNB_model, behav_test, traces_test,
                                                                              behav_max, bin_down)

            # clean the actual and predicted position data using temp_maps by finding the nearest bins in temp_maps
            true_bins = np.array(np.argwhere(temp_maps[:, :, d]))
            for i in range(actual.shape[0]):
                actual_diff, pred_diff = true_bins - np.array(actual[i]), true_bins - np.array(predictions[i])
                actual_norms, pred_norms = np.linalg.norm(actual_diff, axis=1), np.linalg.norm(pred_diff, axis=1)
                actual[i], predictions[i] = true_bins[np.argmin(actual_norms)], true_bins[np.argmin(pred_norms)]
            distances[d, fold] = np.sqrt(err_squared).mean()
            for i, (x, y) in enumerate(actual):
                count_map[x, y] += 1
                err_squared_map[x, y] += err_squared[i]
            for x in range(predictions_dist.shape[0]):
                for y in range(predictions_dist.shape[1]):
                    predictions_dist[x, y] = (count_map + 1).sum() * predictions_dist[x, y] / (count_map + 1)
                    # nan_mask = ~np.isnan(predictions_dist[x, y].flatten())
                    if np.any(predictions_dist[x, y].astype(bool)):
                        smooth_dist = gaussian_filter(predictions_dist[x, y], sigma=1.5)
                        r_, _ = pearsonr(smooth_dist.flatten(), predictions_dist[x, y].flatten())
                        coherence = .5 * np.log((1 + r_) / (1 - r_))
                        coherence_maps[x, y, d] = coherence
                    else:
                        coherence_maps[x, y, d] = np.nan

            mse[:, :, d] = err_squared_map / count_map
            imse[:, :, d] = 1 / err_squared_map

    return distances, np.sqrt(mse), imse, coherence_maps


########################################################################################################################
# Methods to build dataframes for plotting and stats from numpy arrays and nested dictionaries
def get_all_shr_pvals(animals, p):
    '''
    function will collect all pre-calculated p values for every cell across animals into single dataframe
    for subsequent analysis and plotting
    :param animals: list of animals included to collect pre-calculated p values
    :param p: path to pre-calculated p values
    :return: df_pvals: pandas dataframe indicating SHR p value for each cell and day
    '''
    p_results = os.path.join(p, "results")
    group_p_vals = {}
    animal_days = np.zeros(len(animals)) * np.nan
    # use cell num counter for later initialization of the pvals mat across animals and days
    total_cells = 0
    for a, animal in enumerate(animals):
        group_p_vals[animal] = joblib.load(os.path.join(p_results, f'{animal}_shr'))
        total_cells += group_p_vals[animal].shape[0]
        animal_days[a] = group_p_vals[animal].shape[1]
    max_days = animal_days.max()
    p_vals = np.zeros([total_cells, int(max_days)]) * np.nan
    animal_id = np.zeros([total_cells, int(max_days)]) * np.nan
    idx = 0
    for a, animal in enumerate(animals):
        n_cells, n_days = group_p_vals[animal].shape
        p_vals[idx:idx + n_cells, :n_days] = group_p_vals[animal]
        animal_id[idx:idx + n_cells, :n_days] = a
        idx += n_cells
    cols = ['Animal', 'Day', 'Cell', 'SHR']
    df_pvals = pd.DataFrame(columns=cols, data=np.zeros([p_vals.ravel().shape[0], len(cols)])*np.nan)
    idx = 0
    for c in range(p_vals.shape[0]):
        for d in range(p_vals.shape[1]):
           df_pvals.iloc[idx] = pd.Series(np.array([animal_id[c, d], d, c, p_vals[c, d]]))
           idx += 1
    return df_pvals.dropna(axis=0)


def get_all_decoding_within(animals, p):
    p_results = os.path.join(p, "results")
    group_decoding = joblib.load(os.path.join(p_results, "within_decoding"))
    n_animals = len(list(group_decoding.keys()))
    max_days = max([group_decoding[animal]['decoding_error'].shape[0] for animal in list(group_decoding.keys())])
    decoding = np.zeros([n_animals, int(max_days)]) * np.nan
    for a, animal in enumerate(animals):
        n_days = group_decoding[animal]['decoding_error'].shape[0]
        decoding[a, :n_days] = group_decoding[animal]['decoding_error'].mean(1)
    cols = ['Day', 'Animal', 'Error']
    df_decoding = pd.DataFrame(columns=cols, data=np.zeros([decoding.ravel().shape[0], len(cols)])*np.nan)
    idx = 0
    for a in range(decoding.shape[0]):
        for d in range(decoding.shape[1]):
            df_decoding.iloc[idx] = pd.Series(np.array([d, a, decoding[a, d]]))
            idx += 1
    return df_decoding.dropna(axis=0)


########################################################################################################################
# MDS method with fixes to non-metric method
def _smacof_single(dissimilarities, metric=True, n_components=2, init=None,
                   max_iter=300, verbose=0, eps=1e-3, random_state=None):
    """Computes multidimensional scaling using SMACOF algorithm
    Parameters
    ----------
    dissimilarities : ndarray, shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.
    metric : boolean, optional, default: True
        Compute metric or nonmetric SMACOF algorithm.
    n_components : int, optional, default: 2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.
    init : ndarray, shape (n_samples, n_components), optional, default: None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.
    max_iter : int, optional, default: 300
        Maximum number of iterations of the SMACOF algorithm for a single run.
    verbose : int, optional, default: 0
        Level of verbosity.
    eps : float, optional, default: 1e-3
        Relative tolerance with respect to stress at which to declare
        convergence.
    random_state : int, RandomState instance, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term: `Glossary <random_state>`.
    Returns
    -------
    X : ndarray, shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.
    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
    n_iter : int
        The number of iterations corresponding to the best stress.
    """
    dissimilarities = check_symmetric(dissimilarities, raise_exception=True)

    n_samples = dissimilarities.shape[0]
    random_state = check_random_state(random_state)

    sim_flat = ((1 - np.tri(n_samples)) * dissimilarities).ravel()
    sim_flat_w = sim_flat[sim_flat != 0]
    if init is None:
        # Randomly choose initial configuration
        X = random_state.rand(n_samples * n_components)
        X = X.reshape((n_samples, n_components))
    else:
        # overrides the parameter p
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError("init matrix should be of shape (%d, %d)" %
                             (n_samples, n_components))
        X = init

    old_stress = None
    ir = IsotonicRegression()
    for it in range(max_iter):
        # Compute distance and monotonic regression
        dis = euclidean_distances(X)

        if metric:
            disparities = dissimilarities
        else:
            dis_flat = dis.ravel()
            # dissimilarities with 0 are considered as missing values
            dis_flat_w = dis_flat[sim_flat != 0]

            # Compute the disparities using a isotonic regression
            disparities = ir.fit_transform(dissimilarities.ravel(), dis.ravel()).reshape(n_samples, n_samples)
            # disparities = dis_flat.copy()
            # disparities[sim_flat != 0] = disparities_flat
            # sim = sim_flat.reshape((n_samples, n_samples))
            # disparities = disparities.reshape((n_samples, n_samples))
            # disparities[sim_flat.T!=0] = disprities[sim_flat!=0]
            disparities *= np.sqrt((n_samples * (n_samples - 1) / 2) /
                                   (disparities ** 2).sum())

        # Compute stress
        if metric:
            stress = np.sqrt(((dis.ravel() - disparities.ravel()) ** 2).sum() / disparities.ravel().sum())
        else:
            stress = np.sqrt(((dis.ravel() - disparities.ravel()) ** 2).sum() / dis.ravel().sum())

        # Update X using the Guttman transform
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        B = - ratio
        B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
        X = 1. / n_samples * np.dot(B, X)

        dis = np.sqrt((X ** 2).sum(axis=1)).sum()
        if verbose >= 2:
            print('it: %d, stress %s' % (it, stress))
        if old_stress is not None:
            if (old_stress - stress / dis) < eps:
                if verbose:
                    print('breaking at iteration %d with stress %s' % (it,
                                                                       stress))
                break
        old_stress = stress / dis

    return X, stress, it + 1


@_deprecate_positional_args
def smacof(dissimilarities, *, metric=True, n_components=2, init=None,
           n_init=8, n_jobs=None, max_iter=300, verbose=0, eps=1e-3,
           random_state=None, return_n_iter=False):
    """Computes multidimensional scaling using the SMACOF algorithm.
    The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
    multidimensional scaling algorithm which minimizes an objective function
    (the *stress*) using a majorization technique. Stress majorization, also
    known as the Guttman Transform, guarantees a monotone convergence of
    stress, and is more powerful than traditional techniques such as gradient
    descent.
    The SMACOF algorithm for metric MDS can summarized by the following steps:
    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.
    The nonmetric algorithm adds a monotonic regression step before computing
    the stress.
    Parameters
    ----------
    dissimilarities : ndarray, shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.
    metric : boolean, optional, default: True
        Compute metric or nonmetric SMACOF algorithm.
    n_components : int, optional, default: 2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.
    init : ndarray, shape (n_samples, n_components), optional, default: None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.
    n_init : int, optional, default: 8
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress. If ``init`` is
        provided, this option is overridden and a single run is performed.
    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    max_iter : int, optional, default: 300
        Maximum number of iterations of the SMACOF algorithm for a single run.
    verbose : int, optional, default: 0
        Level of verbosity.
    eps : float, optional, default: 1e-3
        Relative tolerance with respect to stress at which to declare
        convergence.
    random_state : int, RandomState instance, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term: `Glossary <random_state>`.
    return_n_iter : bool, optional, default: False
        Whether or not to return the number of iterations.
    Returns
    -------
    X : ndarray, shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.
    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
    n_iter : int
        The number of iterations corresponding to the best stress. Returned
        only if ``return_n_iter`` is set to ``True``.
    Notes
    -----
    "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
    Groenen P. Springer Series in Statistics (1997)
    "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
    Psychometrika, 29 (1964)
    "Multidimensional scaling by optimizing goodness of fit to a nonmetric
    hypothesis" Kruskal, J. Psychometrika, 29, (1964)
    """

    dissimilarities = check_array(dissimilarities)
    random_state = check_random_state(random_state)

    if hasattr(init, '__array__'):
        init = np.asarray(init).copy()
        if not n_init == 1:
            warnings.warn(
                'Explicit initial positions passed: '
                'performing only one init of the MDS instead of %d'
                % n_init)
            n_init = 1

    best_pos, best_stress = None, None

    if effective_n_jobs(n_jobs) == 1:
        for it in range(n_init):
            pos, stress, n_iter_ = _smacof_single(
                dissimilarities, metric=metric,
                n_components=n_components, init=init,
                max_iter=max_iter, verbose=verbose,
                eps=eps, random_state=random_state)
            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = n_iter_
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_smacof_single)(
                dissimilarities, metric=metric, n_components=n_components,
                init=init, max_iter=max_iter, verbose=verbose, eps=eps,
                random_state=seed)
            for seed in seeds)
        positions, stress, n_iters = zip(*results)
        best = np.argmin(stress)
        best_stress = stress[best]
        best_pos = positions[best]
        best_iter = n_iters[best]

    if return_n_iter:
        return best_pos, best_stress, best_iter
    else:
        return best_pos, best_stress


class MDS(BaseEstimator):
    """Multidimensional scaling
    Read more in the :ref:`User Guide <multidimensional_scaling>`.
    Parameters
    ----------
    n_components : int, optional, default: 2
        Number of dimensions in which to immerse the dissimilarities.
    metric : boolean, optional, default: True
        If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.
    n_init : int, optional, default: 4
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress.
    max_iter : int, optional, default: 300
        Maximum number of iterations of the SMACOF algorithm for a single run.
    verbose : int, optional, default: 0
        Level of verbosity.
    eps : float, optional, default: 1e-3
        Relative tolerance with respect to stress at which to declare
        convergence.
    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    random_state : int, RandomState instance, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term: `Glossary <random_state>`.
    dissimilarity : 'euclidean' | 'precomputed', optional, default: 'euclidean'
        Dissimilarity measure to use:
        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset.
        - 'precomputed':
            Pre-computed dissimilarities are passed directly to ``fit`` and
            ``fit_transform``.
    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the position of the dataset in the embedding space.
    stress_ : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import MDS
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = MDS(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)
    References
    ----------
    "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
    Groenen P. Springer Series in Statistics (1997)
    "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
    Psychometrika, 29 (1964)
    "Multidimensional scaling by optimizing goodness of fit to a nonmetric
    hypothesis" Kruskal, J. Psychometrika, 29, (1964)
    """

    @_deprecate_positional_args
    def __init__(self, n_components=2, *, metric=True, n_init=4,
                 max_iter=300, verbose=0, eps=1e-3, n_jobs=None,
                 random_state=None, dissimilarity="euclidean"):
        self.n_components = n_components
        self.dissimilarity = dissimilarity
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def fit(self, X, y=None, init=None):
        """
        Computes the position of the points in the embedding space
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.
        y : Ignored
        init : ndarray, shape (n_samples,), optional, default: None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.
        """
        self.fit_transform(X, init=init)
        return self

    def fit_transform(self, X, y=None, init=None):
        """
        Fit the data from X, and returns the embedded coordinates
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.
        y : Ignored
        init : ndarray, shape (n_samples,), optional, default: None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.
        """
        X = self._validate_data(X)
        if X.shape[0] == X.shape[1] and self.dissimilarity != "precomputed":
            warnings.warn("The MDS API has changed. ``fit`` now constructs an"
                          " dissimilarity matrix from data. To use a custom "
                          "dissimilarity matrix, set "
                          "``dissimilarity='precomputed'``.")

        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_ = X
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix_ = euclidean_distances(X)
        else:
            raise ValueError("Proximity must be 'precomputed' or 'euclidean'."
                             " Got %s instead" % str(self.dissimilarity))

        self.embedding_, self.stress_, self.n_iter_ = smacof(
            self.dissimilarity_matrix_, metric=self.metric,
            n_components=self.n_components, init=init, n_init=self.n_init,
            n_jobs=self.n_jobs, max_iter=self.max_iter, verbose=self.verbose,
            eps=self.eps, random_state=self.random_state,
            return_n_iter=True)

        return self.embedding_
