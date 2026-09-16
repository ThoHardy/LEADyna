# imports
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




##################################### Vizualization Tools ######################################################


# standard colormap
def colormap(cat):
    colormap_dict = {0: (0, 0, 0), 1: (0, 0.25, 1), 2: (0, 0.9375, 1), 3: (0, 0.91, 0.1), 4: (1, 0.6, 0), 5: (1, 0, 0), 6: (0.8, 0, 0)}
    return colormap_dict[cat]


# plot trajectories
def trajectories(time_series, trials=None, avg=False, cursor=None, save_path=None, title=None, figsize=(10,5)):
    '''
    - "trials" is a dict with categories as keys and trial arrays as elements.
    Ex : trials = {0 : [0, 1, 2], 5 : [0, 1, 2, 3, 4]}.
    - "compare" is True for comparing filtered and not filtered trajectories.
    '''
    trials = {0:[],1:[],2:[],3:[],4:[],5:[]} if trials==None else trials
    nb_timepoints = len(time_series[list(time_series.keys())[0]][0])
    times = np.linspace(-500, 2000, nb_timepoints)
    categories = time_series.keys()
    plt.figure(figsize=(10,5))
    for cat in trials.keys() :
        plt.plot(times, np.mean(time_series[cat],0),color=colormap(cat),label=str(cat)) if avg else None
        for traj_nb in trials[cat] :
            plt.plot(times, time_series[cat][traj_nb], linestyle=':' if avg else None, color=colormap(cat) if len(list(trials.keys()))>1 else None)
    if cursor != None:
        xmin, xmax = plt.ylim()
        for value in cursor : 
            plt.plot([value, value], [xmin, xmax], linestyle='--', color='black')
    plt.xlabel('time')
    plt.ylabel('x(t)')
    plt.legend()
    plt.title(title) if title != None else None
    plt.savefig(save_path) if save_path != None else None
    plt.show()


# plot densities around one time-point
def densities(time_series, xlims=None, tlims=None, save_path=None, title=None, figsize=(10,3)):
    
    categories = time_series.keys()
    nb_timepoints = len(time_series[list(categories)[0]][0])
    (t1, t2) = tlims if tlims != None else (0, nb_timepoints)
    time_series = {cat : time_series[cat][:,t1:t2] for cat in categories}
    avgs = {cat : np.mean(time_series[cat],-1).flatten() for cat in categories}
    
    plt.figure(figsize=(10,3))
    for cat in categories :
        sns.kdeplot(data=avgs[cat],label=str(cat),color=colormap(cat))
    plt.xlim(xlims[0], xlims[1]) if xlims!=None else None
    plt.xlabel('x')
    plt.legend()
    plt.title(title) if title != None else None
    plt.savefig(save_path) if save_path != None else None
    plt.show()