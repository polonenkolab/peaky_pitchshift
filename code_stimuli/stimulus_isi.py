# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:01:13 2024

@author: mpolonen
"""
import numpy as np
import pathlib
import scipy.signal as sig
from expyfun.io import read_hdf5, write_hdf5
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.ion()

src = pathlib.Path(__file__).parents[0].resolve().as_posix()
# %% parameters
task = 'peakypitch'
fs = 48e3
dur_stim = 10

stimuli = ['clicks', 'male', 'female']
narrators = ['male', 'female']
f0s = ['low', 'mid', 'high']
ears = ['left', 'right']  # for clicks only

n_stim = len(stimuli)
n_narrs = len(narrators)
n_f0s = len(f0s)
n_ears = len(ears)

n_min_speech = 20.
n_min_clicks = 5.
n_trials_speech = int(n_min_speech * 60 / dur_stim)
n_trials_clicks = int(n_min_clicks * 60 / dur_stim)

# import data
data = read_hdf5(f'{src}/stimuli/clicks_f0rates_regress.hdf5')
pinds = data['pinds']
rates = data['rates']
del data
pinds_clicks = []
for fi in range(n_f0s):
    pinds_clicks.append(pinds[fi][:n_trials_clicks])

pinds_speech = []
for ni, narr in enumerate(narrators):
    pinds_narr = []
    for fi, f0 in enumerate(f0s):
        pinds = read_hdf5(f'{src}/stimuli/{narr}_{f0}f0_regress.hdf5')['pinds']
        pinds_narr.append(pinds[:n_trials_speech])
    pinds_speech.append(pinds_narr)

del pinds, pinds_narr

# %% calculate isi
isi_clicks = []
for fi in range(n_f0s):
    isi = []
    for i in range(n_trials_clicks):
        for ei in range(n_ears):
            isi += list(np.diff(pinds_clicks[fi][i][ei]) / fs)
    isi_clicks.append(np.array(isi))

isi_speech = []
for ni in range(n_narrs):
    isi_narr = []
    for fi in range(n_f0s):
        n_inds = []
        isi = []
        for i in range(n_trials_speech):
            n_inds.append(len(pinds_speech[ni][fi][i]) - 1)
            isi += list(np.diff(pinds_speech[ni][fi][i]) / fs)
        isi_narr.append(np.array(isi))
    isi_speech.append(isi_narr)


# %% filter < 80 ms
isi_speech_filt = []
for ni in range(n_narrs):
    isi = []
    for fi in range(n_f0s):
        dat = isi_speech[ni][fi]
        dat = dat[dat < 80e-3]
        isi.append(dat)
    isi_speech_filt.append(isi)

isi_stim = [isi_clicks, isi_speech_filt[0], isi_speech_filt[1]]
# %% plot histograms
colors = ['tab:blue', 'tab:orange', 'tab:green']
plt.close('all')
fig = plt.figure(figsize=(9, 3))
for si, stim in enumerate(stimuli):
    plt.subplot(1, n_stim, si + 1)
    for fi, f0 in enumerate(f0s):
        plt.hist(isi_stim[si][fi] * 1e3, bins=100, density=True, label=f0,
                 histtype='stepfilled', alpha=0.3, color=colors[fi])
        plt.hist(isi_stim[si][fi] * 1e3, bins=100, density=True,
                 histtype='step', alpha=1, color=colors[fi])
    plt.xticks(np.arange(0, 81, 5))
    plt.yticks(np.arange(0, .5, .1))
    plt.xlim(0, 32)
    plt.ylim(0, 0.35)
    plt.title(stim)
    if si == 0:
        plt.legend(loc='upper right', edgecolor='none', facecolor='none')
    else:
        plt.gca().set_yticklabels([])
fig.text(0.5, 0, 'ISI (ms)', ha='center', va='bottom')
fig.text(0, 0.5, 'Probability density', rotation=90, ha='left', va='center')
plt.tight_layout()
plt.savefig(f'{src}/plots/isi.jpg')

# %% save data
write_hdf5(f'{src}/data_plot/data_isi.hdf5', dict(
    isi_stim=isi_stim,
    isi_clicks=isi_clicks,
    isi_speech=isi_speech,
    stimuli=stimuli,
    narrators=narrators), overwrite=True)
