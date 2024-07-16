# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:14:08 2021

@author: Maddox
"""
import numpy as np
from expyfun.io import read_hdf5, write_hdf5, read_wav

# opath = 'C:/Users/Maddox/Code/Experiments/pip_speech_shift/'
opath = '/mnt/data/peaky_pitch/stimuli/slices/'
click_root = 'clicks_{}.hdf5'
speech_root = '{}_{}f0/broadband/{}{:03}_{}f0_broadband.wav'
rates = [123, 150, 183]
narrators = ['male', 'female']
f0s = ['low', 'mid', 'high']

slice_dur = 10.
n_min_speech = 20
n_trials = int(n_min_speech * 60 / slice_dur)
fs_stim = 48e3

# combine into 10s segments
for ri, rate in enumerate(rates):
    data = read_hdf5(opath + 'clicks/old_1s/' + click_root.format(rate))
    x = data['x'].mean(-2)  # clicks, no bands
    pulses = data['x_pulse'].mean(-2)
    fs = data['fs']

    # slice_dur = 10.
    n_slices = int(x.shape[0] / slice_dur)
    n_samples = int(slice_dur * fs)

    x_10s = np.zeros((n_slices, x.shape[1], x.shape[-1] * slice_dur))
    pulses_10s = np.zeros((n_slices, x.shape[1], x.shape[-1] * slice_dur))
    for si in range(n_slices):
        x_10s[si] = np.concatenate(x[
                si * slice_dur: si * slice_dur + slice_dur], -1)
        pulses_10s[si] = np.concatenate(pulses[
                si * slice_dur: si * slice_dur + slice_dur], -1)
    data['x_10s'] = x_10s
    data['x_pulse_10s'] = pulses_10s
    write_hdf5(opath + 'clicks/' + click_root.format(rate), data,
               overwrite=True)

# combine into 1 file to load
x = np.zeros((len(rates),) + x_10s.shape)
x_pulse = np.zeros((len(rates),) + x_10s.shape)
for ri, rate in enumerate(rates):
    data = read_hdf5(opath + click_root.format(rate))
    x[ri] = data['x_10s']
    x_pulse[ri] = data['x_pulse_10s']
    fs = data['fs']
    del data
write_hdf5(opath + 'clicks/clicks_f0rates_10s.hdf5', dict(
        fs=fs, rates=rates, x_10s=x, x_pulse_10s=x_pulse, stim_dur=slice_dur,
        interleave=True),
    overwrite=True)

# combine speech files into 1 file to load
wavs = np.zeros((len(narrators), len(f0s), n_trials, int(slice_dur * fs_stim)))
wavs.fill(np.nan)
for ni, narr in enumerate(narrators):
    for fi, f0 in enumerate(f0s):
        for ti in range(n_trials):
            wav, fs = read_wav(opath + speech_root.format(
                narr, f0, narr, ti, f0))
            wavs[ni, fi, ti] = wav[0]
            # wavs[ni, fi, ti], fs = read_wav(opath + speech_root.format(
            #     narr, f0, narr, ti, f0))
            assert fs == fs_stim
write_hdf5(opath + 'speech_f0_10s_wavs.hdf5', dict(
    fs=fs, meanf0s=rates, f0s=f0s, narrators=narrators, wavs=wavs,
    n_trials=n_trials, stim_dur=slice_dur), overwrite=True)
