# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:07:42 2024

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
# experiment
task = 'peakypitch'
fs = 10e3
dur_stim = 10

stimuli = ['clicks', 'male', 'female']
narrators = ['male', 'female']
f0s = ['low', 'mid', 'high']
markers = ['mp', 'mjp']
peaks = ['I', 'III', 'V']

n_subs = 15
n_stim = len(stimuli)
n_narrs = len(narrators)
n_f0s = len(f0s)
n_mks = len(markers)
n_pks = len(peaks)

n_min_speech = 20.
n_min_clicks = 5.
n_trials_speech = int(n_min_speech * 60 / dur_stim)
n_trials_clicks = int(n_min_clicks * 60 / dur_stim)

# analysis
tmin = -1
tmax = 1
n_pre = int(-tmin * fs)
n_post = int(tmax * fs)
t = np.arange(-n_pre, n_post) / float(fs)
t_ms = t * 1e3
n_samples = int(n_pre + n_post + dur_stim * fs)

l_freq = 1.
h_freq = 2e3
notch_freq = np.arange(60, 500, 120)
notch_width = 5

n_reps = 2  # odds/evens
# %% functions


def bp_filter(x, lf, hf, order=1, ftype='bandpass', fs_in=fs):
    br, ar = sig.butter(order, np.array([lf, hf]), btype=ftype, fs=fs_in)
    xf = sig.lfilter(br, ar, x)
    return xf


# %% import data
# measured
data = read_hdf5(f'{src}/data_eeg_subs.hdf5')
w_all = data['w_all']
w_reps = data['w_reps']
# w_shuff = data['w_shuff']
# g = data['g']
rates = np.array(data['rates'])
regressors = data['regressors']
n_reg = len(regressors)

# modeled
w_model = np.zeros((n_stim, n_f0s, int(dur_stim * fs)))
for si, stim in enumerate(stimuli):
    for fi, f0 in enumerate(f0s):
        data = read_hdf5(f'{src}/data_model/{stim}_{f0}f0_model.hdf5')
        w_model[si, fi] = data['w_pulses']
w_model = np.concatenate((w_model[..., -n_pre:], w_model[..., :n_post]), -1)
del data

# %% filter
bp1, bp2 = 150, 2000
w_all = bp_filter(w_all, bp1, bp2)
w_reps = bp_filter(w_reps, bp1, bp2)
w_model = bp_filter(w_model, bp1, bp2)

# %% plotting params
ylabs = [r'Amplitude ($\mu$V)', 'Amplitude (AU)']

start, stop = 0.38, 0.88
colors_black = plt.cm.Greys(np.linspace(start, stop, n_f0s))[::-1]
colors_blue = plt.cm.Blues(np.linspace(start, stop, n_f0s))[::-1]
colors_orange = plt.cm.Oranges(np.linspace(start, stop, n_f0s))[::-1]
colors_stim = np.array([colors_black, colors_blue, colors_orange])

shapes_v = ['o', 'x']
shapes_stim = ['o', 'v', '^']
ls_stim = [':', '-', '--']
ms = 4

leg_kwargs = dict(edgecolor='none', facecolor='none',
                  handlelength=1, handletextpad=0.5, labelspacing=0.2)
# %% PEAK PICKING - AUTO
# measured responses
xmin, xmax = 6.5, 10
inds = np.arange(int(xmin * 1e-3 * fs) + n_pre,
                 int(xmax * 1e-3 * fs) + 1 + n_pre)
dat_v = w_all[..., inds]
amp_vmax = dat_v.max(-1)
amp_vmin = dat_v.min(-1)
amp_v = amp_vmax - amp_vmin
lat_v = np.zeros(amp_vmax.shape)
lat_vp = np.zeros(amp_vmin.shape)

for i in range(n_subs):
    for ri in range(n_reg):
        for si in range(n_stim):
            for fi in range(n_f0s):
                lv = inds[np.where(w_all[i, ri, si, fi, inds] == amp_vmax[
                    i, ri, si, fi])[0][0]] - n_pre
                lat_v[i, ri, si, fi] = lv * 1e3 / fs
                lv = inds[np.where(w_all[i, ri, si, fi, inds] == amp_vmin[
                    i, ri, si, fi])[0][0]] - n_pre
                lat_vp[i, ri, si, fi] = lv * 1e3 / fs

# plot subs
plt.close('all')
ylim = np.ceil(w_all.max() * 100) / 100
for i in range(n_subs):
    fig = plt.figure(figsize=(n_stim * 4, 6))
    for si, [stim, col] in enumerate(zip(stimuli, colors_stim)):
        for ri, [reg, ylab] in enumerate(zip(regressors, ylabs)):
            plt.subplot2grid((n_reg, n_stim), (ri, si))
            plt.axvline(0, color='k', ls=':', lw=1)
            plt.axhline(0, color='k', ls=':', lw=1)
            for fi, rate in enumerate(rates):
                plt.plot(t_ms, w_all[i, ri, si, fi], color=col[fi],
                         label=f'{rate} Hz')
                plt.plot(lat_v[i, ri, si, fi], amp_vmax[i, ri, si, fi],
                         shapes_v[0], color=col[fi], ms=ms)
                plt.plot(lat_vp[i, ri, si, fi], amp_vmin[i, ri, si, fi],
                         shapes_v[1], color=col[fi], ms=ms)
            plt.xticks(np.arange(-24, 24, 3))
            plt.yticks(np.arange(-ylim, ylim+1, ylim/3))
            plt.xlim(-4, 16)
            plt.ylim(-ylim, ylim)
            plt.title(f'{stim}: {reg}')
            if si == 0:
                plt.ylabel(ylab)
            else:
                plt.gca().set_yticklabels([])
            if si == 0 and ri == 0:
                plt.legend(facecolor='none', edgecolor='none',
                           loc='upper right')
    fig.text(0.5, 0, 'Latency (ms)', ha='center', va='bottom')
    plt.suptitle(f'sub-{i+1:02d}')
    plt.tight_layout()
    plt.savefig(f'{src}/plots_subs/sub-{i+1:02d}_autopicked.jpg', dpi=300)

# modeled responses
xmin, xmax = 6, 12
inds = np.arange(int(xmin * 1e-3 * fs) + n_pre,
                 int(xmax * 1e-3 * fs) + 1 + n_pre)
dat_v_model = w_model[..., inds]
amp_vmax_model = dat_v_model.max(-1)
amp_vmin_model = dat_v_model.min(-1)
amp_v_model = amp_vmax_model - amp_vmin_model
lat_v_model = np.zeros(amp_vmax_model.shape)
lat_vp_model = np.zeros(amp_vmin_model.shape)

for si in range(n_stim):
    for fi in range(n_f0s):
        lv = inds[np.where(w_model[si, fi, inds] == amp_vmax_model[
            si, fi])[0][0]] - n_pre
        lat_v_model[si, fi] = lv * 1e3 / fs
        lv = inds[np.where(w_model[si, fi, inds] == amp_vmin_model[
            si, fi])[0][0]] - n_pre
        lat_vp_model[si, fi] = lv * 1e3 / fs

plt.close('all')
ylim = np.ceil(np.abs(w_model).max() * 100) / 100
fig = plt.figure(figsize=(n_stim * 4, 3))
for si, [stim, col] in enumerate(zip(stimuli, colors_stim)):
    plt.subplot2grid((1, n_stim), (0, si))
    plt.axvline(0, color='k', ls=':', lw=1)
    plt.axhline(0, color='k', ls=':', lw=1)
    for fi, rate in enumerate(rates):
        plt.plot(t_ms, w_model[si, fi], color=col[fi], label=f'{rate} Hz')
        plt.plot(lat_v_model[si, fi], amp_vmax_model[si, fi],
                 shapes_v[0], color=col[fi], ms=ms)
        plt.plot(lat_vp_model[si, fi], amp_vmin_model[si, fi],
                 shapes_v[1], color=col[fi], ms=ms)
    plt.xticks(np.arange(-24, 24, 3))
    plt.yticks(np.arange(-ylim, ylim+1, ylim/3))
    plt.xlim(-4, 16)
    plt.ylim(-ylim, ylim)
    plt.title(f'{stim}')
    if si == 0:
        plt.ylabel(ylab)
        plt.legend(facecolor='none', edgecolor='none', loc='upper right')
    else:
        plt.gca().set_yticklabels([])
fig.text(0.5, 0, 'Latency (ms)', ha='center', va='bottom')
plt.suptitle(f'modeled responses')
plt.tight_layout()
plt.savefig(f'{src}/plots_subs/modeled_autopicked.jpg', dpi=300)

# %% PEAK PICKING MP - import
regressors = ['pulses', 'anmp']
v_amp = np.zeros((n_mks, n_subs, n_reg, n_stim, n_f0s, n_pks))
v_lat = np.zeros((n_mks, n_subs, n_reg, n_stim, n_f0s, n_pks))
amp_data = []
lat_data = []
labels = []

lat1 = np.zeros((n_subs, n_reg, n_stim, n_f0s))
lat2 = np.zeros((n_subs, n_reg, n_stim, n_f0s))
amp1 = np.zeros((n_subs, n_reg, n_stim, n_f0s))
amp2 = np.zeros((n_subs, n_reg, n_stim, n_f0s))

for mi, marker in enumerate(markers):
    for i in range(n_subs):
        fn = f'{marker}_{task}_sub{i+1:02d}_{int(bp1)}-{int(bp2)}Hz.hdf5'
        data = read_hdf5(f'{src}/peaks/{marker}/{fn}')
        assert data['amplitude'].keys() == data['latency'].keys()
        for item in data['amplitude'].keys():
            labels.append(f'{marker}_{item}')
            amp_data.append(data['amplitude'][item])
            lat_data.append(data['latency'][item])
        for ri, reg in enumerate(regressors):
            for si, stim in enumerate(stimuli):
                for fi, f0 in enumerate(f0s):
                    for pi, peak in enumerate(peaks):
                        key = f'sub{i+1:02d}_{reg}_{stim}_{f0}_{peak}'
                        v_amp[mi, i, ri, si, fi, pi] = data['amplitude'][key]
                        v_lat[mi, i, ri, si, fi, pi] = data['latency'][key]

                        if peak == 'V' and marker == 'mjp':
                            l1 = data['lat1'][key]
                            l2 = data['lat2'][key]
                            idx1 = np.where(np.round(t_ms, 1) ==
                                            np.round(l1, 1))[0][0]
                            idx2 = np.where(np.round(t_ms, 1) ==
                                            np.round(l2, 1))[0][0]
                            lat1[i, ri, si, fi] = l1
                            lat2[i, ri, si, fi] = l2
                            amp1[i, ri, si, fi] = w_all[i, ri, si, fi, idx1]
                            amp2[i, ri, si, fi] = w_all[i, ri, si, fi, idx2]
df = pd.DataFrame(data={'labels': labels, 'amp': amp_data, 'lat': lat_data})
df.to_csv(f'{src}/peaks/peak_data.csv')
df.to_csv(f'{src}/data_plot/peak_data.csv')


# plot subs
plt.close('all')
ylim = np.ceil(w_all.max() * 100) / 100
for i in range(n_subs):
    fig = plt.figure(figsize=(n_stim * 4, 6))
    for si, [stim, col] in enumerate(zip(stimuli, colors_stim)):
        for ri, [reg, ylab] in enumerate(zip(regressors, ylabs)):
            plt.subplot2grid((n_reg, n_stim), (ri, si))
            plt.axvline(0, color='k', ls=':', lw=1)
            plt.axhline(0, color='k', ls=':', lw=1)
            for fi, rate in enumerate(rates):
                plt.plot(t_ms, w_all[i, ri, si, fi], color=col[fi],
                         label=f'{rate} Hz')
                plt.plot(lat1[i, ri, si, fi], amp1[i, ri, si, fi],
                         shapes_v[0], color=col[fi], ms=ms)
                plt.plot(lat2[i, ri, si, fi], amp2[i, ri, si, fi],
                         shapes_v[1], color=col[fi], ms=ms)
            plt.xticks(np.arange(-24, 24, 3))
            plt.yticks(np.arange(-ylim, ylim+1, ylim/3))
            plt.xlim(-4, 16)
            plt.ylim(-ylim, ylim)
            plt.title(f'{stim}: {reg}')
            if si == 0:
                plt.ylabel(ylab)
            else:
                plt.gca().set_yticklabels([])
            if si == 0 and ri == 0:
                plt.legend(facecolor='none', edgecolor='none',
                           loc='upper right')
    fig.text(0.5, 0, 'Latency (ms)', ha='center', va='bottom')
    plt.suptitle(f'sub-{i+1:02d}')
    plt.tight_layout()
    plt.savefig(f'{src}/plots_subs/sub-{i+1:02d}_mppicked.jpg', dpi=300)

# plot wave V peaks with mp picks
plt.close('all')
mk_choose = -1
fig = plt.figure()
for ri, reg in enumerate(regressors):
    plt.subplot2grid((n_reg, 2), (ri, 0))
    for si, stim in enumerate(stimuli):
        plt.errorbar(rates + 2 * si, v_amp.mean(1)[mk_choose, ri, si, :, -1],
                     v_amp.std(1)[mk_choose, ri, si, :, -1] / np.sqrt(n_subs),
                     marker=shapes_stim[si], ls=ls_stim[si], color='k',
                     markerfacecolor='white')
        plt.text(rates[-1] + 2 * si + 4, v_amp.mean(1)[mk_choose, ri, si, -1, -1],
                 stim, color='k', ha='left', va='center')
    plt.xticks(np.arange(120, 190, 30))
    plt.yticks(np.arange(.09, .22, .03))
    plt.xlim(110, 195)
    plt.ylim(.075, .2)
    sns.despine()
    plt.ylabel(ylabs[ri])
for ri, reg in enumerate(regressors):
    plt.subplot2grid((n_reg, 2), (ri, 1))
    for si, stim in enumerate(stimuli):
        plt.errorbar(rates + 2 * si, v_lat.mean(1)[mk_choose, ri, si, :, -1],
                     v_lat.std(1)[mk_choose, ri, si, :, -1] / np.sqrt(n_subs),
                     marker=shapes_stim[si], ls=ls_stim[si], color='k',
                     markerfacecolor='white')
        plt.text(rates[-1] + 2 * si + 4, v_lat.mean(1)[mk_choose, ri, si, -1, -1],
                 stim, color='k', ha='left', va='center')
    plt.xticks(np.arange(120, 190, 30))
    plt.yticks(np.arange(7, 8.3, 0.3))
    plt.xlim(110, 195)
    plt.ylim(7.2, 8.3)
    sns.despine()
    plt.ylabel('Latency (ms)')
fig.text(0.5, 0, 'mean f0 or rate (Hz)', ha='center', va='bottom')
fig.text(0.5, 0.5, f'{regressors[-1]} regressor', ha='center', va='center')
fig.text(0.5, 1, f'{regressors[0]} regressor', ha='center', va='top')
plt.tight_layout()
plt.savefig(f'{src}/plots/wavev_picked.jpg', dpi=300)

# %% COMPARE PEAKS (AUTO VS MP)
lat_min = np.min([v_lat[..., -1].min(), lat_v.min()])
lat_max = np.max([v_lat[..., -1].max(), lat_v.max()])
amp_min = np.min([v_amp[..., -1].min(), amp_v.min()])
amp_max = np.max([v_amp[..., -1].max(), amp_v.max()])

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.plot(np.arange(lat_min, lat_max+.1, .5),
         np.arange(lat_min, lat_max+.1, .5), color='gray', lw=1)
plt.plot(v_lat[0, ..., -1].ravel(), v_lat[1, ..., -1].ravel(), ls='none',
         marker='x', markerfacecolor='none', label='mp time2', color='orange')
plt.plot(v_lat[0, ..., -1].ravel(), lat_v.ravel(), ls='none',
         marker='o', markerfacecolor='none', label='auto picked')
plt.xlim(lat_min, lat_max)
plt.ylim(lat_min, lat_max)
plt.xlabel('MP picked')
plt.ylabel('auto picked')
plt.title('wave V latency')
plt.legend(loc='lower right')
plt.subplot(122)
plt.plot(np.arange(amp_min, amp_max+.01, .03),
         np.arange(amp_min, amp_max+.01, .03), color='gray', lw=1)
plt.plot(v_amp[0, ..., -1].ravel(), v_amp[1, ..., -1].ravel(), ls='none',
         marker='x', markerfacecolor='none', label='mp time2', color='orange')
plt.plot(v_amp[0, ..., -1].ravel(), amp_v.ravel(), ls='none',
         marker='o', markerfacecolor='none', label='auto picked')
plt.xlim(amp_min, amp_max)
plt.ylim(amp_min, amp_max)
plt.xlabel('MP picked')
plt.ylabel('auto picked')
plt.title('wave V amplitude')
plt.tight_layout()
plt.savefig(f'{src}/plots_subs/wavev_picked_auto-mp.jpg', dpi=300)


# %% ADJUSTED MODEL -- MP picked
v_amp_model2 = np.zeros((n_stim, n_f0s, n_pks))
v_lat_model2 = np.zeros((n_stim, n_f0s, n_pks))
amp_data_model2 = []
lat_data_model2 = []
labels_model2 = []

lat1_model2 = np.zeros((n_stim, n_f0s))
lat2_model2 = np.zeros((n_stim, n_f0s))
amp1_model2 = np.zeros((n_stim, n_f0s))
amp2_model2 = np.zeros((n_stim, n_f0s))

data = read_hdf5(f'{src}/data_plot/model_data_all.hdf5')
w_clicks = data['w_model_adj_clicks']
w_speech = data['w_model_adj_speech']
w_all = np.concatenate((w_clicks[None], w_speech), axis=0)
br, ar = sig.butter(1, np.array(
    [150., 2000.]) / (fs / 2.), btype='bandpass')
w_model2 = sig.lfilter(br, ar, w_all)

marker = 'mpolo'
fn = f'{marker}_{task}_model_{int(bp1)}-{int(bp2)}Hz.hdf5'
data = read_hdf5(f'{src}/peaks/{marker}/{fn}')
assert data['amplitude'].keys() == data['latency'].keys()
for item in data['amplitude'].keys():
    labels_model2.append(f'{marker}_{item}')
    amp_data_model2.append(data['amplitude'][item])
    lat_data_model2.append(data['latency'][item])
for si, stim in enumerate(stimuli):
    for fi, f0 in enumerate(f0s):
        for pi, peak in enumerate(peaks):
            key = f'model_{stim}_{f0}_{peak}'
            v_amp_model2[si, fi, pi] = data['amplitude'][key]
            v_lat_model2[si, fi, pi] = data['latency'][key]
            if peak == 'V':
                l1 = data['lat1'][key]
                l2 = data['lat2'][key]
                idx1 = np.where(np.round(t_ms, 1) ==
                                np.round(l1, 1))[0][0]
                idx2 = np.where(np.round(t_ms, 1) ==
                                np.round(l2, 1))[0][0]
                lat1_model2[si, fi] = l1
                lat2_model2[si, fi] = l2
                amp1_model2[si, fi] = w_model2[si, fi, idx1]
                amp2_model2[si, fi] = w_model2[si, fi, idx2]
df = pd.DataFrame(data={'labels': labels_model2, 'amp': amp_data_model2,
                        'lat': amp_data_model2})
df.to_csv(f'{src}/peaks/peak_data_modeladj.csv')


# %% SAVE DATA
write_hdf5(f'{src}/data_plot/data_peaks.hdf5', dict(
    v_amp_auto=amp_v,
    v_lat_auto=lat_v,
    v_amp_auto_model=amp_v_model,
    v_lat_auto_model=lat_v_model,
    v_amp_pick=v_amp,
    v_lat_pick=v_lat,
    v_lat_model=v_lat_model2,
    v_amp_model=v_amp_model2,
    lat1=lat1,
    lat2=lat2,
    amp1=amp1,
    amp2=amp2,
    lat1_model2=lat1_model2,
    lat2_model2=lat2_model2,
    amp1_model2=amp1_model2,
    amp2_model2=amp2_model2,
    stimuli=stimuli,
    f0s=f0s,
    rates=rates,
    marker='mp'), overwrite=True)
