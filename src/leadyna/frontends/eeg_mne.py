import numpy as np
import math
import matplotlib.pyplot as plt
import os
import scipy.io
from scipy.signal import butter, filtfilt
import warnings
import pandas
import seaborn as sns
import warnings
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import math
from scipy.stats import zscore
from scipy.signal import savgol_filter
import seaborn as sns
from scipy.stats import pearsonr
import mne



def STG(data_ref, tmin, tmax, substract_pattern=None, sort_by_snr=True, decimate_factor=5):
    '''
    Return Segmented Temporal Generalization analysis for a classifier trained in the (tmin, tmax) time window.
    If substract_pattern == (tmin2, tmax2), the encoding pattern in this window will be removed from the whole time-series via projection in the corresponding hyperplane. 
    '''

    # Load and crop/decimate epochs
    epochs = (
        mne.read_epochs(data_ref, preload=True, verbose=False)
        .decimate(decimate_factor)
        .crop(tmin=tmin/1000, tmax=tmax/1000, include_tmax=True, verbose=False)
    )

    max_cat = int(np.max(np.unique(epochs.metadata['snr'])))

    # Select present vs absent stim epochs
    stim_pres_epochs = epochs[str(max_cat)]
    time_series = {cat: [] for cat in range(max_cat)} if sort_by_snr else []
    stim_abs_epochs = epochs['1']
    combined_epochs = mne.concatenate_epochs([stim_abs_epochs, stim_pres_epochs], verbose=False)

    # Make sure metadata is available
    if combined_epochs.metadata is None or 'blocknumber' not in combined_epochs.metadata.columns:
        raise ValueError("Epochs must have metadata with a 'blocknumber' column.")

    blocks_train = combined_epochs.metadata['blocknumber'].values
    blocks_test = epochs.metadata['blocknumber'].values
    unique_blocks = np.unique(blocks_test)


    # Loop over blocks for leave-one-block-out CV
    for block in unique_blocks:
        train_mask = blocks_train != block
        test_mask = blocks_test == block

        # Training data
        data = combined_epochs[train_mask].get_data()
        if substract_pattern != None:
            t1, t2 = substract_pattern
            w = data[combined_epochs[train_mask].metadata['snr'] == max_cat, :, t1:t2].mean(0).mean(1) - data[combined_epochs[train_mask].metadata['snr'] == 1, :, t1:t2].mean(0).mean(1) # compute pattern
            w_norm2 = np.dot(w, w)
            coeff = np.einsum('ijk,j->ik', data, w) / w_norm2
            coeff = coeff[:, None, :]  # reshape for broadcasting
            data -= coeff * w[:, None] # Remove projection
        
        X_train = data.transpose(0, 2, 1)
        y_train = np.repeat(
            np.where(combined_epochs[train_mask].events[:, 2] == 1, 0, 1),
            X_train.shape[1]
        )
        X_train = X_train.reshape(-1, X_train.shape[-1])  # n_channels inferred from data (was hard-coded 64)

        # Define pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler(with_mean=True)),
            ('classifier', LogisticRegression(solver='liblinear', penalty='l2'))
        ])

        pipeline.fit(X_train, y_train)

        # Test data: compute decision function for each trial in the held-out block
        test_epochs = mne.read_epochs(data_ref, preload=True, verbose=False).decimate(decimate_factor)
        data = test_epochs[test_mask].get_data()
        if substract_pattern != None:
            t1, t2 = substract_pattern
            w = data[test_epochs[test_mask].metadata['snr'] == max_cat, :, t1:t2].mean(0).mean(1) - data[test_epochs[test_mask].metadata['snr'] == 1, :, t1:t2].mean(0).mean(1) # compute pattern
            w_norm2 = np.dot(w, w)
            coeff = np.einsum('ijk,j->ik', data, w) / w_norm2
            coeff = coeff[:, None, :]  # reshape for broadcasting
            data -= coeff * w[:, None] # Remove projection
        
        if sort_by_snr:
            for cat in range(max(list(time_series.keys()))+1):
                try:
                    data_this_snr = data[test_epochs[test_mask].metadata['snr'] == cat+1]
                    for trial in data_this_snr:
                        ts = pipeline.decision_function(trial.T)
                        time_series[cat].append(ts)
                except KeyError:
                    continue
        else:
            for trial in data:
                ts = pipeline.decision_function(trial.T)
                time_series.append(ts)
            

    # Convert lists to arrays, demean and normalize relative to category 0
    if sort_by_snr:
        time_series = {cat: np.array(series) for cat, series in time_series.items()}
        predictors_demean = {cat: time_series[cat] - np.mean(time_series[0], 0) for cat in time_series.keys()}
        predictors = {int(cat): predictors_demean[cat] / np.std(predictors_demean[0]) for cat in predictors_demean.keys()}
    else:
        time_series = np.array(time_series)
        resting_state_trials = time_series[epochs.metadata['snr'] == 1]
        
        predictors_demean = time_series - np.mean(resting_state_trials, 0)
        predictors = predictors_demean / np.std(predictors_demean, 0)

    del epochs, test_epochs, combined_epochs, stim_pres_epochs, stim_abs_epochs

    return predictors