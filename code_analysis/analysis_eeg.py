# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:24:37 2024

@author: mpolonen
"""

import numpy as np
import mne
import pathlib
from glob import glob
import scipy.signal as sig
from scipy.fftpack import fft, ifft
from expyfun.io import read_hdf5, write_hdf5
from expyfun import binary_to_decimals
import matplotlib.pyplot as plt
from mne.filter import resample
plt.ion()

src = pathlib.Path(__file__).parents[0].resolve().as_posix()
# %% parameters
# experiment
task = 'peakypitch'

fs_stim = 48e3
fs = 10e3

dur_stim = 10
dur_stamp = 9.98

stimuli = ['clicks', 'male', 'female']
narrators = ['male', 'female']
f0s = ['low', 'mid', 'high']
ears = ['left', 'right']

n_subs = 15
n_ch = 2
n_stim = len(stimuli)
n_narrs = len(narrators)
n_f0s = len(f0s)
n_ears = len(ears)

n_min_speech = 20.
n_min_clicks = 5.

n_trials_speech = int(n_min_speech * 60 / dur_stim)
n_trials_clicks = int(n_min_clicks * 60 / dur_stim)
n_trials_total = n_trials_clicks * n_f0s + n_trials_speech * n_narrs * n_f0s

n_bits_stim = int(np.ceil(np.log2(n_stim)))
n_bits_narr = int(np.ceil(np.log2(n_narrs)))
n_bits_f0 = int(np.ceil(np.log2(n_f0s)))
n_bits_trial = int(np.ceil(np.log2(np.max(
    [n_trials_speech, n_trials_clicks]))))

# analysis
event_id = {'New Segment/': 0, 'Stimulus/S  1': 1, 'Stimulus/S  2': 2,
            'Stimulus/S  4': 4, 'Stimulus/S  8': 8}
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


def derive_responses(xi, yi, stim_type='speech', shuffled=False,
                     regressor='pulses'):
    # dimensions: n_tokens, n_ears, n_ch, n_samples
    n_fft = mne.filter.next_fast_len(xi.shape[-1])
    x_fft = fft(xi, n=n_fft)
    y_fft = fft(yi, n=n_fft)
    y_fft[..., :1] = 0

    if not regressor == 'pulses':
        x_fft = np.exp(1j * np.angle(x_fft))  # phase only anm regressor
        y_fft = y_fft[:, None]  # add another dimension for pos/neg
        yi = yi[:, None]

    if shuffled:
        n_tr = x_fft.shape[0]
        # shuffle by 1 trial; chose -2 b/c pre/post eeg
        x_fft = np.array([x_fft[i % n_tr - 2] for i in range(n_tr)])

    # trial weights
    tw = 1. / np.var(yi, axis=-1, keepdims=True)
    tw /= tw.sum(0, keepdims=True)

    if stim_type == 'speech':
        w = 1e6 * ifft((np.conj(x_fft) * y_fft * tw).sum(0) /
                       (np.conj(x_fft) * x_fft).mean(0)).real
    elif stim_type == 'clicks':
        w = 1e6 * ifft((np.conj(x_fft) * y_fft * tw).sum(0)).real
        n_x = np.sum(xi != 0, axis=-1, keepdims=True).mean(0)
        w /= n_x
        w = w.mean(-3)  # avg ear
    else:
        assert False
    # concatenate and mean across ch
    w = np.concatenate((w[..., -n_pre:], w[..., :n_post]), -1).mean(-2)

    if not regressor == 'pulses':
        # note: Tong & Ross tested the delay by cross-correlating the
        # click train and the ANM output of the click train
        # Maximized at 2.75 ms (ANM reg already had 1 ms shift) -- so 3.75 ms
        # makes sense given wave I latencies from eLife paper
        w = np.roll(w, int(2.75 * 1e-3 * fs))
        w = w.mean(-2)  # avg by pos/neg

    return w


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


# %% load stimuli (eeg-bids dataset formatting)
print('loading speech regressors')
pinds_speech = []
anm_speech = np.zeros((n_narrs, n_f0s, n_trials_speech, 2, int(dur_stim * fs)))
for ni, narr in enumerate(narrators):
    pinds_narr = []
    for fi, f0 in enumerate(f0s):
        pinds_f0 = []
        for ti in range(n_trials_speech):
            data = read_hdf5(
                f'{src}/stimuli/{narr}_{f0}/{narr}_{f0}_{ti:03d}_regress.hdf5')
            pinds_f0.append(data['pinds'])
            anm_speech[ni, fi, ti] = resample(data['anm'], down=fs_stim/fs)
        pinds_narr.append(pinds_f0)
    pinds_speech.append(pinds_narr)
del pinds_f0, pinds_narr, data

print('loading clicks regressors')
rates = []
pinds_clicks = []
anm_clicks = np.zeros((n_f0s, n_trials_clicks, 2, n_ears, int(dur_stim * fs)))
for fi, f0 in enumerate(f0s):
    pinds_f0 = []
    for ti in range(n_trials_clicks):
        data = read_hdf5(
            f'{src}/stimuli/clicks_{f0}/clicks_{f0}_{ti:03d}_regress.hdf5')
        pinds_f0.append(data['pinds'])
        anm_clicks[fi, ti] = resample(data['anm'], down=fs_stim/fs)
        if ti == 0:
            rates.append(data['rate'])
    pinds_clicks.append(pinds_f0)
del pinds_f0, data

# # %% load stimuli
# pinds_speech = []
# anm_speech = np.zeros((n_narrs, n_f0s, n_trials_speech, 2, int(dur_stim * fs)))
# print('loading speech regressors')
# for ni, narr in enumerate(narrators):
#     pinds_narr = []
#     for fi, f0 in enumerate(f0s):
#         data = read_hdf5(f'{src}/stimuli/{narr}_{f0}f0_regress.hdf5')
#         pinds_narr.append(data['pinds'])
#         anm_speech[ni, fi] = resample(data['anm'], down=fs_stim/fs)
#     pinds_speech.append(pinds_narr)
# del pinds_narr, data

# print('loading clicks regressors')
# data = read_hdf5(f'{src}/stimuli/clicks_f0rates_regress.hdf5')
# pinds_clicks = data['pinds']
# anm_clicks = resample(data['anm'], down=fs_stim/fs)
# rates = data['rates']
# %% organize data (pre-allocate memory)
w_pt = np.zeros((n_subs, n_stim, n_f0s, n_pre + n_post))
w_pt_shuffled = np.zeros((n_subs, n_stim, n_f0s, n_pre + n_post))
w_pt_reps = np.zeros((n_subs, n_stim, n_f0s, n_reps, n_pre + n_post))

w_ap = np.zeros((n_subs, n_stim, n_f0s, n_pre + n_post))
w_ap_shuffled = np.zeros((n_subs, n_stim, n_f0s, n_pre + n_post))
w_ap_reps = np.zeros((n_subs, n_stim, n_f0s, n_reps, n_pre + n_post))

# %% load eeg and pre-process
for si in range(n_subs):
    print(f'\n== sub {si + 1:03d} ==\n')
    print('loading eeg data')
    fn = f'sub-{si+1:02d}/eeg/sub-{si+1:02d}_task-{task}_eeg.vhdr'
    raw = mne.io.brainvision.read_raw_brainvision(
        f'{src}/data_raw_bids/{fn}', preload=True)
    raw._data -= raw._data[..., [0]]
    events = mne.events_from_annotations(raw, event_id)[0]
    assert (raw.info['sfreq'] == fs)

    # filter the eeg
    raw._data = bp_filter(raw._data, l_freq, h_freq)
    for nf in notch_freq:
        bn, an = sig.iirnotch(nf / (raw.info['sfreq'] / 2.),
                              float(nf) / notch_width)
        raw._data = sig.lfilter(bn, an, raw._data)

    # correct for tubing length
    tube_len_meters = 11. * 25.4 * 1e-3
    tube_delay = tube_len_meters / 343.
    tube_delay_len = int(np.round(tube_delay * fs))
    events[:, 0] += tube_delay_len

    # get the right ends figured out; should handle missing start/end triggers
    starts = events[events[:, -1] == 1]
    ends = events[events[:, -1] == 2]
    nom_length = np.median(np.diff(starts[:, 0]))
    all_length = ends[:, [0]] - starts[:, [0]].T
    len_trial = all_length.ravel()[(all_length.ravel() > 0) &
                                   (all_length.ravel() < nom_length)]

    # calculate adjustment for different clocks
    clock_diff = dur_stamp * fs - len_trial
    clock_adjust = (dur_stamp * fs / len_trial).mean()
    fs_adjust = clock_adjust * fs

    # process triggers to get event info
    start_inds = np.array(np.where(events[:, -1] == 1))[0]
    stop_inds = np.array(np.where(events[:, -1] == 2))[0]

    n_bits_stamp = np.sum([n_bits_stim, n_bits_f0, n_bits_trial])
    bits_epoch = [events[start + 1:start + n_bits_stamp + 1, 2] >> 3
                  for start in start_inds]

    stims = np.array([binary_to_decimals(be[
        :n_bits_stim], n_bits_stim)[0] for be in bits_epoch])
    shifts = np.array([binary_to_decimals(be[
        n_bits_stim:n_bits_stim + n_bits_f0], n_bits_f0)[0]
        for be in bits_epoch])
    trials = np.array([binary_to_decimals(be[
        -n_bits_trial:], n_bits_trial)[0] for be in bits_epoch])
    assert len(stims) == len(shifts) == len(trials) == n_trials_total

    event_info = np.zeros((starts.shape)).astype(int)
    event_info[:, 0] = stims
    event_info[:, 1] = shifts
    event_info[:, 2] = trials

    stim_events = starts.copy()
    stim_events[:, -1] = stims
    stim_events[:, -2] = shifts

    print('resampling anm regressors')
    anm_speech_rs = resample(anm_speech, down=fs/fs_adjust)
    anm_clicks_rs = resample(anm_clicks, down=fs/fs_adjust)

    print('deriving waveforms')
    for i in range(n_stim):
        print(f'  {stimuli[i]}')
        for fi in range(n_f0s):
            print(f'  -- {f0s[fi]}')
            events = stim_events[(stim_events[:, -1] == i) &
                                 (stim_events[:, -2] == fi)]
            tokens = event_info[(event_info[:, 0] == i) &
                                (event_info[:, 1] == fi)][:, -1]
            ep = mne.Epochs(raw, events, tmin=tmin, tmax=dur_stim + tmax-1/fs,
                            preload=True, proj=None)
            # baseline=None, detrend=None
            if i > 0:
                stim = 'speech'
                # y_speech[si, i-1, fi] = ep._data
                y = ep._data
                x_pt = np.zeros((n_trials_speech, n_samples))
                x_ap = np.zeros((n_trials_speech, 2, n_samples))
                for ti, tok in enumerate(tokens):
                    pinds = pinds_speech[i-1][fi][tok]
                    x_pt[ti, ((pinds / fs_stim / clock_adjust - tmin)
                              * fs).astype(int)] = 1.
                    x_ap[ti, :, n_pre:n_pre + anm_speech_rs.shape[-1]
                         ] = anm_speech_rs[i-1, fi, tok]
            else:
                stim = 'clicks'
                # y_clicks[si, fi] = ep._data
                y = ep._data[..., None, :, :]  # add for ear
                x_pt = np.zeros((n_trials_clicks, n_ears, n_samples))
                x_ap = np.zeros((n_trials_clicks, 2, n_ears, n_samples))
                for ti, tok in enumerate(tokens):
                    for ei in range(n_ears):
                        pinds = pinds_clicks[fi][tok][ei]
                        x_pt[ti, ei, ((pinds / fs_stim / clock_adjust - tmin)
                                      * fs).astype(int)] = 1.
                    x_ap[ti, :, :, n_pre:n_pre + anm_clicks_rs.shape[-1]
                         ] = anm_clicks_rs[fi, tok]

            x_pt = x_pt[..., None, :]  # add for ch
            x_ap = x_ap[..., None, :]

            w_pt[si, i, fi] = derive_responses(x_pt, y, stim_type=stim)
            w_pt_shuffled[si, i, fi] = derive_responses(
                x_pt, y, stim_type=stim, shuffled=True)
            w_pt_reps[si, i, fi] = np.array([derive_responses(
                x_pt[ri::n_reps], y[ri::n_reps], stim_type=stim)
                for ri in range(n_reps)])

            w_ap[si, i, fi] = derive_responses(
                x_ap, y, stim_type=stim, regressor='anmp')
            w_ap_shuffled[si, i, fi] = derive_responses(
                x_ap, y, stim_type=stim, shuffled=True, regressor='anmp')
            w_ap_reps[si, i, fi] = np.array([derive_responses(
                x_ap[ri::n_reps], y[ri::n_reps], stim_type=stim,
                regressor='anmp') for ri in range(n_reps)])

# %% normalize anmp magnitude
inds_norm = np.arange(n_pre, n_pre + int(20e-3 * fs))

pt = w_pt[..., inds_norm].std(-1).mean(0).mean(-1)
pt = np.array([pt[0], pt[1:].mean(0), pt[1:].mean(0)])
ap = w_ap[..., inds_norm].std(-1).mean(0).mean(-1)
ap = np.array([ap[0], ap[1:].mean(0), ap[1:].mean(0)])

g = pt / ap

w_ap_norm = np.zeros(w_ap.shape)
w_ap_shuffled_norm = np.zeros(w_ap_shuffled.shape)
w_ap_reps_norm = np.zeros(w_ap_reps.shape)
for i in range(n_stim):
    w_ap_norm[:, i] = w_ap[:, i] * g[i]
    w_ap_shuffled_norm[:, i] = w_ap_shuffled[:, i] * g[i]
    w_ap_reps_norm[:, i] = w_ap_reps[:, i] * g[i]

regressors = ['pulses', 'anm-phase']
n_reg = len(regressors)

w_all = np.moveaxis(np.array([w_pt, w_ap_norm]), 0, 1)
w_shuff = np.moveaxis(np.array([w_pt_shuffled, w_ap_shuffled_norm]), 0, 1)
w_reps = np.moveaxis(np.array([w_pt_reps, w_ap_reps_norm]), 0, 1)

# %% save data
data = dict(
    fs=fs,
    w_pt=w_pt,
    w_pt_shuffled=w_pt_shuffled,
    w_pt_reps=w_pt_reps,
    w_ap=w_ap,
    w_ap_shuffled=w_ap_shuffled,
    w_ap_reps=w_ap_reps,
    w_all=w_all,
    w_shuff=w_shuff,
    w_reps=w_reps,
    g=g,
    stimuli=stimuli,
    narrators=narrators,
    f0s=f0s,
    regressors=regressors,
    rates=rates,
    dur_stim=dur_stim,
    t_ms=t_ms,
    t=t,
    n_pre=n_pre,
    n_post=n_post
)

write_hdf5(f'{src}/data_eeg_subs.hdf5', data, overwrite=True)

# %% filter
bp1, bp2 = 150, 2000
w_all = bp_filter(w_all, bp1, bp2)
w_shuff = bp_filter(w_shuff, bp1, bp2)
w_reps = bp_filter(w_reps, bp1, bp2)

# %% plotting params
ylabs = [r'Potential ($\mu$V)', 'Potential (AU)']

start, stop = 0.42, 0.99
colors_black = plt.cm.Greys(np.linspace(start, stop, n_f0s))[::-1]
colors_blue = plt.cm.Blues(np.linspace(start, stop, n_f0s))[::-1]
colors_orange = plt.cm.Oranges(np.linspace(start, stop, n_f0s))[::-1]
colors_stim = np.array([colors_black, colors_blue, colors_orange])

# %% plot subs
plt.close('all')
ylim = np.ceil(w_all.max() * 100) / 100
for i in range(n_subs):
    fig = plt.figure(figsize=(n_stim * 4, 6))
    for si, [stim, col] in enumerate(zip(stimuli, colors_stim)):
        for ri, [reg, ylab] in enumerate(zip(regressors, ylabs)):
            plt.subplot2grid((n_reg, n_stim), (ri, si))
            plt.axvline(0, color='k', ls=':', lw=1)
            plt.axhline(0, color='k', ls=':', lw=1)
            [plt.plot(t_ms, w_all[i, ri, si, fi], color=col[fi],
                      label=f'{rate} Hz') for fi, rate in enumerate(rates)]
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
    plt.savefig(f'{src}/plots_subs/sub-{i+1:02d}.jpg', dpi=300)

# %% plot means
ylim = 0.1
fig = plt.figure(figsize=(n_stim * 4, 6))
for si, [stim, col] in enumerate(zip(stimuli, colors_stim)):
    for ri, [reg, ylab] in enumerate(zip(regressors, ylabs)):
        plt.subplot2grid((n_reg, n_stim), (ri, si))
        plt.axvline(0, color='k', ls=':', lw=1)
        plt.axhline(0, color='k', ls=':', lw=1)
        [error_area(t_ms, w_all[:, ri, si, fi], color=col[fi],
                    label=f'{rate} Hz') for fi, rate in enumerate(rates)]
        plt.xticks(np.arange(-24, 24, 3))
        plt.yticks(np.arange(-ylim, ylim+1, ylim/2))
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
plt.tight_layout()
plt.savefig(f'{src}/plots_subs/grand_avg.jpg', dpi=300)

# %%
ylim = 0.1
fig = plt.figure(figsize=(n_stim * 4, 6))
for fi, rate in enumerate(rates):
    for ri, [reg, ylab] in enumerate(zip(regressors, ylabs)):
        plt.subplot2grid((n_reg, n_f0s), (ri, fi))
        plt.axvline(0, color='k', ls=':', lw=1)
        plt.axhline(0, color='k', ls=':', lw=1)
        [error_area(t_ms, w_all[:, ri, si, fi], color=colors_stim[si, fi],
                    label=stim) for si, stim in enumerate(stimuli)]
        plt.xticks(np.arange(-24, 24, 3))
        plt.yticks(np.arange(-ylim, ylim+1, ylim/2))
        plt.xlim(-4, 16)
        plt.ylim(-ylim, ylim)
        plt.title(f'{rate} Hz: {reg}')
        if fi == 0:
            plt.ylabel(ylab)
        else:
            plt.gca().set_yticklabels([])
        if fi == 0 and ri == 0:
            plt.legend(facecolor='none', edgecolor='none',
                       loc='upper right')
fig.text(0.5, 0, 'Latency (ms)', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(f'{src}/plots_subs/grand_avg2.jpg', dpi=300)
