# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:07:31 2024

@author: mpolonen
"""

from sklearn.model_selection import ParameterGrid
import numpy as np
import pathlib
from expyfun.io import read_hdf5, write_hdf5
import amp_models
import mne
from scipy.fftpack import fft, ifft
import scipy.signal as sig
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
plt.ion()

src = pathlib.Path(__file__).parents[0].resolve().as_posix()

# %% params
task = 'peakypitch'
fn_speech = '{}{}_{}f0_peaky_regress.hdf5'
fn_clicks = 'clicks_f0rates_regress.hdf5'

narrators = ['male', 'female']
f0s = ['low', 'mid', 'high']
waves = ['w1', 'w3', 'w5']
n_narr = len(narrators)
n_f0s = len(f0s)
n_waves = len(waves)

fs_audio = int(48e3)
fs_eeg = int(10e3)
shift_ms = 1
stim_db = 65
dur_stim = 10

n_min_speech = 20
n_min_clicks = 5
n_trials_speech = int(n_min_speech * 60 / dur_stim)
n_trials_clicks = int(n_min_clicks * 60 / dur_stim)
n_samples = int(dur_stim * fs_eeg)

# model scaling
# n_cfs = 43
# model_scale = 401 / n_cfs

# response params
t_min = -1
t_max = 1
n_pre = int(-t_min * fs_eeg)
n_post = int(t_max * fs_eeg)
t = np.arange(-n_pre, n_post) / float(fs_eeg)
t_ms = t * 1e3

# filter params
lf = 150.
hf = 2000.

data_save = dict()

# %% create model eeg for clicks
data = read_hdf5(f'{src}/stimuli/{fn_clicks}')
audio_clicks = data['x']
pinds_clicks = data['pinds']
n_ears = audio_clicks.shape[-2]

pulses_clicks = np.zeros((n_trials_clicks, n_f0s, n_ears, n_samples))
model_eeg_clicks = np.zeros((n_trials_clicks, n_f0s, n_ears, n_samples))
model_waves_clicks = np.zeros(
    (n_trials_clicks, n_waves, n_f0s, n_ears, n_samples))
for fi, f0 in enumerate(f0s):
    print(f'\n == clicks: {f0} rate == \n')
    for ti in range(n_trials_clicks):
        print(f'\n-- trial {ti + 1} / {n_trials_clicks} --\n')
        for ei in range(n_ears):
            audio = audio_clicks[fi, ti, ei]
            pinds = [(pi * float(fs_eeg) / float(fs_audio)).astype(int)
                     for pi in pinds_clicks[fi][ti][ei]]
            model, model_scale = amp_models.model_abr(
                audio, fs_audio, fs_eeg, stim_db, return_flag='135abr')
            pulses_clicks[ti, fi, ei, pinds] = 1.
            model_eeg_clicks[ti, fi, ei] = model['abr']
            model_waves_clicks[ti, :, fi, ei] = np.array(
                [model[wi] for wi in waves])

data_save['pulses_clicks'] = pulses_clicks
data_save['model_eeg_clicks'] = model_eeg_clicks
data_save['model_waves_clicks'] = model_waves_clicks
data_save['model_scale'] = model_scale
write_hdf5(f'{src}/data_plot/model_data_all.hdf5', data_save, overwrite=True)
# %% create model eeg for speech
pulses_speech = np.zeros((n_trials_speech, n_narr, n_f0s, n_samples))
model_eeg_speech = np.zeros((n_trials_speech, n_narr, n_f0s, n_samples))
model_waves_speech = np.zeros(
    (n_trials_speech, n_waves, n_narr, n_f0s, n_samples))
for ni, narr in enumerate(narrators):
    for fi, f0 in enumerate(f0s):
        print(f'\n == {narr}: {f0} f0 == \n')
        data = read_hdf5(f'{src}/stimuli/{narr}_{f0}f0_regress.hdf5')
        audio = data['x']
        pinds_speech = data['pinds']
        assert fs_audio == data['fs']
        for ti in range(n_trials_speech):
            print(f'\n-- trial {ti + 1} / {n_trials_speech} --\n')
            pinds = [(pi * float(fs_eeg) / float(fs_audio)).astype(int)
                     for pi in pinds_speech[ni][fi][ti]]
            model, model_scale = amp_models.model_abr(
                audio, fs_audio, fs_eeg, stim_db, return_flag='135abr')
            pulses_speech[ti, ni, fi, pinds] = 1.
            model_eeg_speech[ti, ni, fi] = model['abr']
            model_waves_speech[ti, :, ni, fi] = np.array(
                [model[wi] for wi in waves])

data_save['pulses_speech'] = pulses_speech
data_save['model_eeg_speech'] = model_eeg_speech
data_save['model_waves_speech'] = model_waves_speech
data_save['model_scale'] = model_scale
write_hdf5(f'{src}/data_plot/model_data_all.hdf5', data_save, overwrite=True)
# %% functions


def derive_modeled_abrs(xi, yi, stim_type='speech'):
    # dimensions: have n_tokens in first dim
    n_fft = mne.filter.next_fast_len(xi.shape[-1])
    x_fft = fft(xi, n=n_fft)
    y_fft = fft(yi, n=n_fft)

    if stim_type == 'speech':
        w = 1e6 * ifft((np.conj(x_fft) * y_fft).mean(0) /
                       (np.conj(x_fft) * x_fft).mean(0)).real
    elif stim_type == 'clicks':
        w = 1e6 * ifft((np.conj(x_fft) * y_fft).mean(0)).real
        n_x = np.sum(xi != 0, axis=-1, keepdims=True).mean(0)
        w /= n_x
        w = w.mean(-2)  # avg ear
    else:
        assert False
    # concatenate and mean across ch
    w = np.concatenate((w[..., -n_pre:], w[..., :n_post]), -1)
    return w


def bp_filter(x, lf, hf, order=1, ftype='bandpass', fs_in=fs_eeg):
    br, ar = sig.butter(order, np.array([lf, hf]), btype=ftype, fs=fs_in)
    xf = sig.lfilter(br, ar, x)
    return xf


def abr_params(x, y, params, inds, fs=fs_eeg):
    w1_shift = int(fs * params['shift1'] * 1e-3)
    w3_shift = int(fs * params['shift3'] * 1e-3)
    w5_shift = int(fs * params['shift5'] * 1e-3)
    w1 = np.roll(y[:, 0] * params['m1'], w1_shift, axis=-1)
    w3 = np.roll(y[:, 1] * params['m3'], w3_shift, axis=-1)
    w5 = np.roll(y[:, 2] * params['m5'], w5_shift, axis=-1)
    for i in range(y.shape[0]):
        for ii in range(y.shape[-2]):
            w1[i, ii, :w1_shift] = w1[i, ii, w1_shift+1]
            w3[i, ii, :w3_shift] = w3[i, ii, w3_shift+1]
            w5[i, ii, :w5_shift] = w5[i, ii, w5_shift+1]
    eeg = w1 + w3 + w5
    abr = derive_modeled_abrs(x, eeg, 'clicks')
    abr_filt = bp_filter(abr, lf, hf)[inds]
    return abr_filt


# %% derive modeled abrs
w_model_clicks = derive_modeled_abrs(
    pulses_clicks, model_eeg_clicks, stim_type='clicks')
w_model_speech = derive_modeled_abrs(pulses_speech, model_eeg_speech)

data_save['w_model_clicks'] = w_model_clicks
data_save['w_model_speech'] = w_model_speech

# w_model_clicks *= model_scale
# w_model_speech *= model_scale

# %% import click measured data
w_all = read_hdf5(f'{src}/data_plot/data_eeg_subs.hdf5')['w_all']
w_clicks = w_all.mean(0)[0, 0]  # mean subs, pulses reg, click
w_speech = w_all.mean(0)[0, 1:]

# %% plot
inds_abr = np.where((t_ms >= 0) & (t_ms <= 13))[0]
abr_model_clicks_low = bp_filter(w_model_clicks[0], lf, hf)[inds_abr]
abr_click_low = bp_filter(w_clicks[0], lf, hf)[inds_abr]
tms = t_ms[inds_abr]

plt.close('all')
plt.figure()
plt.axhline(0, color='k', lw=1, ls=':', alpha=0.3)
plt.plot(tms, abr_click_low, color='k', label='measured')
plt.plot(tms, abr_model_clicks_low, color='orange', label='modeled')
plt.legend(loc='upper right')
plt.title('Clicks low rate')
plt.ylabel('Amplitude')
plt.xlabel('Latency (ms)')
plt.tight_layout()

# y = model_waves_clicks[:, :, 0]
# x = pulses_clicks[:, 0]
# %% grid without multiplying by model_scale = 401 / len(cf)
# scales = np.arange(0.4, 0.6, 0.1)
# scales = np.arange(1.0, 6.1, .1)
shifts = np.arange(0, 0.1, .1)

# scales = np.arange(0, 400, 100)
# shifts = np.arange(0, 1, 1)
param_grid = {'m1': np.arange(1.2, 1.6, .1),
              'm3': np.arange(1.8, 2.3, .1),
              'm5': np.arange(4.9, 5.4, .1),
              'shift1': np.arange(0, 0.4, 0.1),
              'shift3': np.arange(0.4, 0.7, 0.1),
              'shift5': np.arange(0.3, 0.6, 0.1)}
grid = ParameterGrid(param_grid)
inds_abr = np.where((t_ms >= 0) & (t_ms <= 13))[0]

results = np.zeros([len(grid), len(inds_abr)])
results = Parallel(n_jobs=-1)([delayed(abr_params)(
    pulses_clicks[:, 0], model_waves_clicks[:, :, 0],
    params, inds_abr) for params in grid])

results = np.array(results)
data_save['grid_results'] = results
write_hdf5(f'{src}/data_plot/model_data_all.hdf5', data_save, overwrite=True)
# %%
inds_diff = np.where(tms <= 8.5)[0]
diffs = np.abs(results[:, inds_diff] - abr_click_low[None, inds_diff]).sum(-1)
# diffs = np.abs(results - abr_click_low[None]).sum(-1)
diff_min = np.where(diffs == diffs.min())[0][0]
abr_model_opt_params = grid[diff_min]
abr_model_opt = results[diff_min]

# plt.close('all')
# plt.figure()
# plt.axhline(0, color='k', lw=1, ls=':', alpha=0.3)
# plt.plot(tms, abr_click_low, color='k', label='measured')
plt.plot(tms, abr_model_opt, color='r', label='normalized')
plt.legend(loc='upper right')
# plt.title('Clicks low rate')
# plt.ylabel('Amplitude')
# plt.xlabel('Latency (ms)')
# plt.tight_layout()


# %% plot
inds_abr = np.where((t_ms >= 0) & (t_ms <= 13))[0]
abr_model_clicks_low = bp_filter(w_model_clicks[0], lf, hf)[inds_abr]
abr_click_low = bp_filter(w_clicks[0], lf, hf)[inds_abr]
tms = t_ms[inds_abr]

plt.close('all')
plt.figure()
plt.axhline(0, color='k', lw=1, ls=':', alpha=0.3)
plt.plot(tms, abr_click_low, color='k', label='measured')
plt.plot(tms, abr_model_clicks_low, color='orange', label='modeled')
plt.plot(tms, abr_model_opt, color='teal', label='normalized')
plt.legend(loc='upper right')
plt.title('Clicks low rate')
plt.ylabel('Amplitude')
plt.xlabel('Latency (ms)')
plt.tight_layout()
plt.savefig(f'{src}/plots/model_adjust.jpg')
# %%

# a, b, c = range(3), range(3), range(3)
# def my_func(x, y, z):
#     return (x + y + z) / 3.0, x * y * z, max(x, y, z)
# grids = numpy.vectorize(my_func)(*numpy.ix_(a, b, c))
# mean_grid, product_grid, max_grid = grids
