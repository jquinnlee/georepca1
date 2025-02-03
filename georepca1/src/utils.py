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