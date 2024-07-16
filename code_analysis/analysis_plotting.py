# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:08:26 2024

@author: mpolonen
"""
import numpy as np
import pathlib
import scipy.signal as sig
from expyfun.io import read_hdf5, write_hdf5
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import copy
plt.ion()

src = pathlib.Path(__file__).parents[0].resolve().as_posix()
# %% parameters
# experiment
task = 'peakypitch'
fs = 10e3
dur_stim = 10

stimuli = ['Clicks', 'Male', 'Female']
narrators = ['Male', 'Female']
f0s = ['low', 'mid', 'high']
markers = ['mp']
peaks = ['I', 'III', 'V']
regressors = ['pulses', 'anm-p']
labels = ['Previous work', 'Current work']

n_subs = 15
n_stim = len(stimuli)
n_narrs = len(narrators)
n_f0s = len(f0s)
n_mks = len(markers)
n_pks = len(peaks)
n_reg = len(regressors)

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


def error_area(x, y, color, alpha=0.3, line_alpha=1, do_std=False, **kwargs):
    ym = y.mean(0)
    e = y.std(0)
    if not do_std:
        e /= np.sqrt(y.shape[0])
    plt.fill_between(
        x, ym - e, ym + e,
        facecolor=color, color=color, alpha=alpha, edgecolor='none')
    plt.plot(x, ym.T, color=color, alpha=line_alpha, **kwargs)


# %% import data
print('loading data')
# previous exp
data = read_hdf5(f'{src}/data_plot/exp_elife2_data.hdf5')
w_elife = data['w_bbn_all']

# example audio
data = read_hdf5(f'{src}/data_plot/audio_ex.hdf5')
audios = ['Unaltered', 'Peaky']
n_audios = len(audios)
audio_tinds = data['t_inds']
audio_tms = data['t_ms']
audio_wavs = np.array([data['x'][audio_tinds], data['x_single'][audio_tinds]])
audio_pulses = data['pulses'][audio_tinds]
audio_fs = data['fs']
t_spec = 60e-3
ff, tt, spec_x = sig.spectrogram(
    audio_wavs[0], fs=audio_fs, window='hann',
    nperseg=int(audio_fs * t_spec), noverlap=int(audio_fs * t_spec * 0.9))
ff, tt, spec_xps = sig.spectrogram(
    audio_wavs[1], fs=audio_fs, window='hann',
    nperseg=int(audio_fs * t_spec), noverlap=int(audio_fs * t_spec * 0.9))
spectograms = np.array([spec_x, spec_xps])

# isi
data = read_hdf5(f'{src}/data_plot/data_isi.hdf5')
isi_stim = data['isi_stim']

# measured
data = read_hdf5(f'{src}/data_plot/data_eeg_subs.hdf5')
w_all = data['w_all'][:, 0]  # decided not to show anm-phase regressor abrs
w_shuff = data['w_shuff'][:, 0]
w_reps = data['w_reps'][:, 0]
g = data['g']
rates = np.array(data['rates'])
regressors = data['regressors']
n_reg = len(regressors)

# modeled
# data = read_hdf5(f'{src}/data_plot/data_model.hdf5')
# w_model = data['w']
# w_model = np.concatenate((w_model[..., -n_pre:], w_model[..., :n_post]), -1)

data = read_hdf5(f'{src}/data_plot/model_data_all.hdf5')
model_scale = 401 / 43
w_model_orig = model_scale * np.concatenate((
    data['w_model_orig_clicks'][None], data['w_model_orig_speech']), axis=0)
w_model_opt = np.concatenate((
    data['w_model_adj_clicks'][None], data['w_model_adj_speech']), axis=0)
# params_model_orig=abr_model_orig_params,
# params_model_adj=abr_model_opt_params3,
# params_model_final=abr_model_params

# peaks
data = read_hdf5(f'{src}/data_plot/data_peaks.hdf5')
v_lat = data['v_lat_pick'][:, :, 0]  # decided not to use anm-phase regr abrs
v_amp = data['v_amp_pick'][:, :, 0]
v_lat_auto = data['v_lat_auto'][:, 0]
v_amp_auto = data['v_amp_auto'][:, 0]
# v_lat_model = data['v_lat_auto_model']
# v_amp_model = data['v_amp_auto_model']
lat1 = data['lat1'][:, 0]
lat2 = data['lat2'][:, 0]
amp1 = data['amp1'][:, 0]
amp2 = data['amp2'][:, 0]
v_lat_model = data['v_lat_model'][..., -1]
v_amp_model = data['v_amp_model'][..., -1]
lat1_model = data['lat1_model2']
lat1_model = data['lat1_model2']
lat2_model = data['lat2_model2']
amp1_model = data['amp1_model2']
amp2_model = data['amp2_model2']
del data

# %% filter
bp1, bp2 = 150, 2000
w_elife = bp_filter(w_elife, bp1, bp2)

w_all = bp_filter(w_all, bp1, bp2)
w_shuff = bp_filter(w_shuff, bp1, bp2)
w_reps = bp_filter(w_reps, bp1, bp2)

w_model_orig = bp_filter(w_model_orig, bp1, bp2)
w_model_opt = bp_filter(w_model_opt, bp1, bp2)

# %% calculations
# natural f0s (eLife and current)
w_natural = np.array([w_elife, w_all[:, [1, -1], [0, -1]]], dtype=object)
inds_norm = np.arange(n_pre, n_pre + int(20e-3 * fs))
g = (w_natural[1][..., inds_norm].std(-1).mean(0).mean(0) /
     w_natural[0][..., inds_norm].std(-1).mean(0).mean(0))

w_natural_norm = copy.deepcopy(w_natural)
w_natural_norm[0] *= g

# correlations of m/f vs e/o
corr_inds = np.arange(int(2e-3 * fs) + n_pre, int(12e-3 * fs) + 1 + n_pre)
w_corr = np.array([np.moveaxis(w_reps[:, 1:].mean(1), -2, 1),
                   w_reps[:, 1:].mean(-2)])[..., corr_inds]
types = ['evens/odds', 'male/female']
params = ['r', 'p']
n_types = len(types)
n_params = len(params)
corrs = np.zeros((n_types, n_subs, n_f0s, n_params))
for ti in range(n_types):
    for si in range(n_subs):
        for fi in range(n_f0s):
            corrs[ti, si, fi] = stats.pearsonr(
                w_corr[ti, si, 0, fi], w_corr[ti, si, 1, fi])
wxcns = np.zeros((n_f0s, n_params))
for fi in range(n_f0s):
    wxcns[fi] = stats.wilcoxon(corrs[0, :, fi, 0], corrs[1, :, fi, 0])
wxcns[:, -1] = fdrcorrection(wxcns[:, -1])[-1]

for ti, types in enumerate(['m/f', 'e/o']):
    for fi, f0 in enumerate(f0s):
        print(f'{types}: {f0}')
        print(np.percentile(corrs[ti, :, fi, 0],
              np.array([25, 50, 75])).round(3))

# natural f0s wave V ratios for m/f current vs previous
abr_inds = np.where((t_ms > 2) & (t_ms < 12))[0]
v_factor_natural = []
for w in w_natural:
    vmax = w[..., abr_inds].max(-1)
    vmin = w[..., abr_inds].min(-1)
    amp = vmax - vmin
    amp_ratio = amp[:, 0] / amp[:, 1]
    v_factor_natural.append(amp_ratio)
    print(amp_ratio.mean(0).round(2))
    print(np.round(amp_ratio.std(0) / np.sqrt(w.shape[0]), 2))

stats.shapiro(v_factor_natural[0])  # normal
stats.shapiro(v_factor_natural[1])  # not normal
for i, w in enumerate(v_factor_natural):
    plt.subplot(1, 2, i+1)
    plt.hist(w)
mwu_naturalV = stats.mannwhitneyu(v_factor_natural[0], v_factor_natural[1])
# auto-pick modeled abr wave V peaks
# autopick_inds = np.arange(int(4e-3 * fs) + n_pre, int(12e-3 * fs) + 1 + n_pre)
# v_amp1_model = w_model_opt[..., autopick_inds].max(-1)
# v_amp2_model = w_model_opt[..., autopick_inds].min(-1)
# v_lat1_model = np.zeros((n_stim, n_f0s))
# v_lat2_model = np.zeros((n_stim, n_f0s))
# for si in range(n_stim):
#     for fi in range(n_f0s):
#         lv = autopick_inds[np.where(w_model_opt[
#             si, fi, autopick_inds] == v_amp1_model[si, fi])[0][0]] - n_pre
#         v_lat1_model[si, fi] = lv * 1e3 / fs
#         lv = autopick_inds[np.where(w_model_opt[
#             si, fi, autopick_inds] == v_amp2_model[si, fi])[0][0]] - n_pre
#         v_lat2_model[si, fi] = lv * 1e3 / fs
# v_amp_model = v_amp1_model - v_amp2_model
# v_lat_model = v_lat1_model

for fi in range(n_f0s):
    print(stats.mannwhitneyu(isi_stim[-2][fi] * 1e3, isi_stim[-1][fi] * 1e3))
# isi_stim[si][fi] * 1e3

# %% plotting params
ylabs_w = ['Potential (μV)', 'Potential (AU)']
ylabs_peaks = ['Amplitude (μV)', 'Amplitude (AU)']
ylab_units = ['(μV)', '(AU)']
ylab_pot = 'Potential (μV)'

start, stop = 0.38, 0.88
colors_black = plt.cm.Greys(np.linspace(start, stop, n_f0s))[::-1]
colors_blue = plt.cm.Blues(np.linspace(start, stop, n_f0s))[::-1]
colors_orange = plt.cm.Oranges(np.linspace(start, stop, n_f0s))[::-1]
colors_red = plt.cm.Reds(np.linspace(start, stop, n_f0s))[::-1]
colors_purple = plt.cm.Purples(np.linspace(start, stop, n_f0s))[::-1]
colors_greens = plt.cm.Greens(np.linspace(start, stop, n_f0s))[::-1]
colors_stim = np.array([colors_black, colors_blue, colors_orange])

start, stop = 0.15, 0.75
start, stop = 0.12, 0.78
colors = plt.cm.viridis(np.linspace(start, stop, n_f0s))
plt.clf()
for i in range(3):
    plt.plot(i, i, marker='o', ls='none', color=colors[i], ms=10)

shapes_v = ['o', 'x']
shapes_stim = ['o', 'v', '^']
ls_stim = [':', '-', '--']
ls_stim = ['-', '-', '-']
alphas_wavs = [1, 0.5]


leg_kwargs = dict(edgecolor='none', facecolor='none',
                  handlelength=1, handletextpad=0.5, labelspacing=0.2)
axline_kwargs = dict(color='k', ls=':', lw=0.8, alpha=0.2)

# sns.set_context('paper', rc={'lines.linewidth': 1, 'font.family': 'Arial',
#                              'text.usetex': False, 'pdf.fonttype': 42})
# colw = 3
dpi = 300

# inches max for JASA-EL
colw = 6.68
colh = 8.25

path_plots = f'{src}/plots_final/'
sns.set_context('paper', rc={
    'font.size': 9.4,  # default 9.6
    'axes.labelsize': 9.4,
    'axes.titlesize': 9.4,
    'legend.title_fontsize': 9.4,
    'savefig.dpi': 300,
    'xtick.major.size': 3.8,
    'ytick.major.size': 3.8,
})

# plt.rcParams['font.size'] = plt.rcParams['']
# plt.rcParams['axes.labelsize'] = 8
# plt.rcParams['axes.titlesize'] = 8.5

assert False
# %% FIG 1 -  stim & elife/current abr m/f
plt.close('all')
# fig = plt.figure(figsize=(colw * 2, colw * 1.5))
fig = plt.figure(figsize=(colw, colh/2.2))
xmin, xmax = 0.1, 0.4
ymax = 0.075
for ai, audio in enumerate(audios):
    # pressure waveforms
    plt.subplot2grid((n_audios+1, 2), (ai, 0))
    plt.plot(audio_tms, audio_wavs[ai], color='k', alpha=alphas_wavs[ai], lw=1)
    plt.text(xmax, ymax, f'{audio} speech', color='k', alpha=alphas_wavs[ai],
             ha='right', va='top')
    plt.xticks(np.arange(0, 1.2, 0.1))
    plt.xlim(xmin, xmax)
    plt.ylim(-0.06, ymax+.01)
    plt.gca().set_yticks([])
    if ai == 0:
        plt.gca().spines[:].set_visible(False)
        plt.gca().set_xticks([])
    else:
        plt.gca().spines[['left', 'right', 'top']].set_visible(False)
        plt.plot(audio_tms, audio_pulses * 0.02 - 0.07, color='k',
                 alpha=alphas_wavs[ai], lw=1)
        plt.text(xmax, -0.07 + 0.026, 'Pulse regressor', color='k',
                 alpha=alphas_wavs[ai], ha='right', va='bottom')
        # plt.xlabel('Time (s)')
    # spectrograms
    plt.subplot2grid((n_audios+1, 2), (ai, 1))
    plt.imshow(10 * np.log10(spectograms[ai]), aspect='auto', origin='lower',
               clim=[-120, -60], extent=[tt[0], tt[-1], ff[0], ff[-1]],
               cmap='viridis')
    plt.ylim(0, 10000)
    plt.yticks(np.arange(0, 12000, 2000), np.arange(0, 12, 2))
    plt.xticks(np.arange(0, 1.2, 0.1))
    plt.xlim(xmin, xmax)
    plt.gca().spines[['right', 'top']].set_visible(False)
    if ai == 0:
        plt.gca().set_xticklabels([])
    # else:
        # plt.xlabel('Time (s)')

fig.text(0.505, 0.66, 'Frequency (kHz)', ha='left', va='center', rotation=90)
fig.text(0, 0.66, 'Amplitude (AU)', ha='left', va='center', rotation=90)
fig.text(0.25, 0.33, 'Time (s)', ha='center', va='bottom')
fig.text(0.75, 0.33, 'Time (s)', ha='center', va='bottom')
fig.text(0.75, 0.66, 'Yellow = higher amplitude', ha='center', va='bottom')

# eLife m/f abrs
xmin, xmax = -2, 13
ylims = [[-.13, .13], [-.075, .074]]
# ymin, ymax = -.13, .13
for i, [lab, w, ylim] in enumerate(zip(labels, w_natural, ylims)):
    plt.subplot2grid((n_audios+1, 2), (2, i))
    plt.axvline(0, **axline_kwargs)
    plt.axhline(0, **axline_kwargs)
    for ni, narr in enumerate(narrators):
        error_area(t_ms, w[:, ni], color=colors[ni+1], label=narr)
    plt.xticks(np.arange(-24, 24, 4))
    plt.yticks(np.arange(-.12, .13, .06))
    plt.xlim(xmin, xmax)
    plt.ylim(ylim[0], ylim[-1])
    plt.text(0.5, ylim[-1], lab, ha='left', va='top')
    if i == 0:
        plt.legend(loc=(0.7, 0.70), **leg_kwargs)
    # else:
    #     plt.gca().set_yticklabels([])
    plt.gca().spines[['right', 'top']].set_visible(False)
# labels
fig.text(0, 0.18, ylab_pot, rotation=90, ha='left', va='center')
fig.text(0.5, 0, 'Latency (ms)', ha='center', va='bottom')

[fig.text(0, yi, lab, fontweight='bold', ha='left', va='top')
 for yi, lab in zip([0.98, 0.35], ['a', 'b'])]
plt.tight_layout()

plt.subplots_adjust(wspace=0.2, hspace=0.63)  # .18, .42
plt.savefig(f'{path_plots}Figure1_audioex_mfabrs.jpg', dpi=dpi)

# %% FIG 2 -  stim ISI
plt.close('all')
fig = plt.figure(figsize=(colw/2.5, colh/2.5))
for fi, f0 in enumerate(rates):
    plt.subplot(n_stim, 1, fi + 1)
    for si, stim in enumerate(stimuli):
        plt.hist(isi_stim[si][fi] * 1e3, bins=100, density=True, label=stim,
                 histtype='stepfilled', alpha=0.3, color=colors[si])
        plt.hist(isi_stim[si][fi] * 1e3, bins=100, density=True,
                 histtype='step', alpha=.7, color=colors[si])
    plt.xticks(np.arange(0, 81, 5))
    plt.yticks(np.arange(0, .5, .1))
    plt.xlim(0, 27)
    plt.ylim(0, 0.35)
    # plt.text(27 / 2, 0.30, f'{f0} Hz', ha='center', va='top')
    plt.text(26.5, 0.01, f'{f0} Hz', ha='right', va='bottom')
    if fi == 0:
        plt.legend(loc='upper right', **leg_kwargs)
    if fi < n_stim - 1:
        plt.gca().set_xticklabels([])
    sns.despine()
fig.text(0.5, 0, 'Inter-stimulus interval (ms)', ha='center', va='bottom')
fig.text(0, 0.5, 'Probability density', rotation=90, ha='left', va='center')
plt.tight_layout()
plt.subplots_adjust(hspace=0.25)
plt.savefig(f'{path_plots}/Figure2_isi_byf0.jpg', dpi=dpi)

# %% FIG 3 - measured ABRs and wave V peaks
plt.close('all')

# fig = plt.figure(figsize=(colw * 2, colw * 1.5))
fig = plt.figure(figsize=(colw, colh/2.3))
# GA waveforms
xmin, xmax = -2, 13
ymin, ymax = -.10, .09
for fi, f0 in enumerate(rates):
    # plt.subplot2grid((7, 6), (0, fi * 2), colspan=2, rowspan=4)
    plt.subplot2grid((7, 12), (0, fi * 4), colspan=4, rowspan=4)
    plt.axvline(0, **axline_kwargs)
    plt.axhline(0, **axline_kwargs)
    for si, stim in enumerate(stimuli):
        error_area(t_ms, w_all[:, si, fi],
                   color=colors[si], label=stim, ls=ls_stim[si])
    plt.xticks(np.arange(-24, 24, 4))
    plt.yticks(np.arange(-.12, .13, .04))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # plt.title(f'{f0} Hz')
    plt.text(xmin + (xmax - xmin) / 2, ymax,
             f'{f0} Hz', ha='center', va='center')
    if fi == 0:
        plt.legend(loc='upper left', **leg_kwargs)
    else:
        plt.gca().set_yticklabels([])
    sns.despine()
fig.text(0.5, 3.1/7, 'Latency (ms)', ha='center', va='top')
fig.text(0, 0.75, ylab_pot, rotation=90, ha='left', va='center')

ms = 6
ymin, ymax = 0.05, 0.2
# plt.subplot2grid((7, 6), (4, 0), colspan=3, rowspan=3)
plt.subplot2grid((7, 12), (4, 0), colspan=5, rowspan=3)
for si, stim in enumerate(stimuli):
    plt.errorbar(rates + 2 * si, v_amp.mean(0).mean(0)[si, :, -1],
                 v_amp.mean(0).std(0)[si, :, -1] / np.sqrt(n_subs),
                 marker=shapes_stim[si], ls=ls_stim[si], color=colors[si],
                 markerfacecolor='white', ms=ms)
    # plt.text(rates[-1] + 2 * si + 4, v_amp.mean(0).mean(0)[si, -1, -1],
    #          stim, color=colors[si], ha='left', va='center')
plt.xticks(rates)
plt.yticks(np.arange(.06, .22, .03))
plt.xlim(120, 190)
plt.ylim(ymin, ymax)
sns.despine()
ymin, ymax = 7.1, 8.3
# plt.subplot2grid((7, 6), (4, 3), colspan=3, rowspan=3)
plt.subplot2grid((7, 12), (4, 6), colspan=5, rowspan=3)
for si, stim in enumerate(stimuli):
    plt.errorbar(rates + 2 * si, v_lat.mean(0).mean(0)[si, :, -1],
                 v_lat.mean(0).std(0)[si, :, -1] / np.sqrt(n_subs),
                 marker=shapes_stim[si], ls=ls_stim[si], color=colors[si],
                 markerfacecolor='white', ms=ms)
    # plt.text(rates[-1] + 2 * si + 4, v_lat.mean(0).mean(0)[si, -1, -1],
    #          stim, color=colors[si], ha='left', va='center')
plt.xticks(rates)
plt.yticks(np.arange(7, 8.3, 0.3))
plt.xlim(120, 190)
plt.ylim(ymin, ymax)
sns.despine()

fig.text(0.5, 0, 'Mean f0 or rate (Hz)', ha='center', va='bottom')
fig.text(0.495, 0.25, f'Latency (ms)', rotation=90, ha='center', va='center')
fig.text(0, 0.25, f'Amplitude (μV)', rotation=90, ha='left', va='center')
[fig.text(0, yi, lab, fontweight='bold', ha='left', va='top')
 for yi, lab in zip([3.2/7, 0.98], ['b', 'a'])]
[fig.text(xi, yi, lab, ha='left', va='center', color=colors[si])
 for si, [xi, yi, lab] in enumerate(zip([0.421, 0.43, 0.41],
                                        [0.31, 0.21, 0.14], stimuli))]
[fig.text(xi, yi, lab, ha='left', va='center', color=colors[si])
 for si, [xi, yi, lab] in enumerate(zip([0.88, 0.89, 0.9],
                                        [0.19, 0.31, 0.36], stimuli))]
# plt.suptitle('Meausured ABRs')
plt.tight_layout()
plt.subplots_adjust(wspace=0.7)


plt.savefig(f'{path_plots}/Figure3_abrs_measured_Vpeaks.jpg', dpi=300)

# %% FIG 4 - modeled ABRs
plt.close('all')

# fig = plt.figure(figsize=(colw * 2, colw * 1.5))
fig = plt.figure(figsize=(colw, colh/2.3))
# modeled waveforms
xmin, xmax = -2, 13
ymin, ymax = -.10, .09
for fi, f0 in enumerate(rates):
    # plt.subplot2grid((7, 6), (0, fi * 2), colspan=2, rowspan=4)
    plt.subplot2grid((7, 12), (0, fi * 4), colspan=4, rowspan=4)
    plt.axvline(0, **axline_kwargs)
    plt.axhline(0, **axline_kwargs)
    for si, stim in enumerate(stimuli):
        plt.plot(t_ms, w_model_opt[si, fi],
                 color=colors[si], label=stim, ls=ls_stim[si])
    plt.xticks(np.arange(-24, 24, 4))
    plt.yticks(np.arange(-.12, .13, .04))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # plt.title(f'{f0} Hz')
    plt.text(xmin + (xmax - xmin) / 2, ymax,
             f'{f0} Hz', ha='center', va='center')
    if fi == 0:
        plt.legend(loc='upper left', **leg_kwargs)
    else:
        plt.gca().set_yticklabels([])
    sns.despine()
fig.text(0.5, 3.1/7, 'Latency (ms)', ha='center', va='top')
fig.text(0, 0.73, 'Modeled potential (μV)',
         rotation=90, ha='left', va='center')

ms = 6
ymin, ymax = 0.05, 0.2
# plt.subplot2grid((7, 6), (4, 0), colspan=3, rowspan=3)
plt.subplot2grid((7, 12), (4, 0), colspan=5, rowspan=3)
for si, stim in enumerate(stimuli):
    plt.plot(rates + 2 * si, v_amp_model[si], marker=shapes_stim[si],
             ls=ls_stim[si], color=colors[si], markerfacecolor='white', ms=ms)
    # plt.text(rates[-1] + 2 * si + 4, v_amp_model[si, -1], stim,
    #           color=colors[si], ha='left', va='center')
plt.xticks(rates)
plt.yticks(np.arange(.06, .22, .03))
plt.xlim(120, 190)
plt.ylim(ymin, ymax)
sns.despine()
ymin, ymax = 7.1, 8.3
# plt.subplot2grid((7, 6), (4, 3), colspan=3, rowspan=3)
plt.subplot2grid((7, 12), (4, 6), colspan=5, rowspan=3)
for si, stim in enumerate(stimuli):
    plt.errorbar(rates + 2 * si, v_lat_model[si], marker=shapes_stim[si], ms=ms,
                 ls=ls_stim[si], color=colors[si], markerfacecolor='white')
    # plt.text(rates[-1] + 2 * si + 4, v_lat_model[si, -1],
    #          stim, color=colors[si], ha='left', va='center')
plt.xticks(rates)
plt.yticks(np.arange(7, 8.3, 0.3))
plt.xlim(120, 190)
plt.ylim(ymin, ymax)
sns.despine()
fig.text(0.5, 0, 'Mean f0 or rate (Hz)', ha='center', va='bottom')
fig.text(0.495, 0.25, 'Latency (ms)', rotation=90, ha='center', va='center')
fig.text(0, 0.25, r'Amplitude (μV)', rotation=90, ha='left', va='center')
[fig.text(0, yi, lab, fontweight='bold', ha='left', va='top')
 for yi, lab in zip([3.2/7, 0.98], ['b', 'a'])]
[fig.text(xi, yi, lab, ha='left', va='center', color=colors[si])
 for si, [xi, yi, lab] in enumerate(zip([0.42, 0.39, 0.41],
                                        [0.26, 0.19, 0.16], stimuli))]
[fig.text(xi, yi, lab, ha='left', va='center', color=colors[si])
 for si, [xi, yi, lab] in enumerate(zip([0.88, 0.89, 0.9],
                                        [0.15, 0.28, 0.33], stimuli))]

plt.tight_layout()
plt.subplots_adjust(wspace=0.7)
plt.savefig(f'{path_plots}/Figure4_abrs_modeled_Vpeaks.jpg', dpi=300)

# %% SUPPL FIG 1 - abr picked peaks
plt.close('all')
abr_inds = np.where((t_ms > 2) & (t_ms < 12))[0]
w_all[..., abr_inds].max()
w_all[..., abr_inds].min()
xmin, xmax = -2, 14
ymin, ymax = -.15, .15
nrow = 5
ncol = 3
ms = 3.5
ls_stim
for fi, f0 in enumerate(rates):
    fig = plt.figure(figsize=(colw, colh))
    for i in range(n_subs):
        plt.subplot(nrow, ncol, i + 1)
        plt.axvline(0, **axline_kwargs)
        plt.axhline(0, **axline_kwargs)
        for si, stim in enumerate(stimuli):
            plt.plot(t_ms, w_all[i, si, fi],
                     color=colors[si], label=stim, ls=ls_stim[si])
            plt.plot(lat1[i, si, fi], amp1[i, si, fi], color=colors[si],
                     marker=shapes_v[0], ms=ms)
            plt.plot(lat2[i, si, fi], amp2[i, si, fi], color=colors[si],
                     marker=shapes_v[-1], ms=ms)
        plt.xticks(np.arange(-24, 24, 4))
        plt.yticks(np.arange(-1, 1, .1))
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.text(xmax, ymax, f'sub {i+1:02d}', ha='right', va='top')
        if i == 0:
            plt.legend(loc=(0, 0), ncol=1, **leg_kwargs)
        if i % ncol > 0:
            plt.gca().set_yticklabels([])
        if i // ncol < 4:
            plt.gca().set_xticklabels([])
        sns.despine()
    fig.text(0.5, 0, 'Latency (ms)', ha='center', va='bottom')
    fig.text(0, 0.5, ylab_pot, rotation=90, ha='left', va='center')
    plt.suptitle(f'{f0} Hz', x=0.5, y=1)
    plt.tight_layout()
    plt.savefig(
        f'{path_plots}/Supp_Figure{fi+1}_indivABRs_{f0}Hz.jpg', dpi=300)
