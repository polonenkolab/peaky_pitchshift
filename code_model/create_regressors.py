# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 19:13:57 2024

@author: mpolonen
"""

import numpy as np
import pathlib
from expyfun.io import read_hdf5, write_hdf5
import amp_models
# from mne.filter import resample

# %%  params
src = pathlib.Path(__file__).parents[0].resolve().as_posix()
fn_speech = '{}{}_{}f0_peaky_reduced.hdf5'
fn_clicks = 'clicks_f0rates_10s.hdf5'

narrators = ['male', 'female']
f0s = ['low', 'mid', 'high']
fs_audio = 48e3
fs_eeg = 10e3
shift_ms = 1
stim_db = 65
dur_stim = 10
n_samples = int(dur_stim * fs_audio)

n_min_speech = 20
n_min_clicks = 5
n_trials_speech = int(n_min_speech * 60 / dur_stim)
n_trials_clicks = int(n_min_clicks * 60 / dur_stim)

# %% create anm regressor for speech
for narr in narrators:
    for f0 in f0s:
        print(f'\n == {narr}: {f0} f0 == \n')
        for ti in range(n_trials_speech):
            print(f'\n-- trial {ti + 1} / {n_trials_speech} --\n')
            fn = f'{narr}{ti:03d}_{f0}f0_peaky_reduced.hdf5'
            data = read_hdf5(f'{src}/stimuli/speech/{narr}_{f0}f0/{fn}')
            pinds = data['pulse_inds'][0]
            audio = data['x_play_single']
            fs = data['fs']
            f0_info = data['f0info']
            assert n_samples == audio.shape[0]
            anm = np.zeros((2, n_samples))  # 2: pos, neg
            anm[0] = amp_models.anm(audio, fs_audio, stim_db)
            anm[1] = amp_models.anm(-audio, fs_audio, stim_db)
            # anm_rs = resample(anm, down=fs_audio/fs_eeg)
            # chose not to resample so all audio is in its original fs_audio
            data = dict(pinds=pinds, x=audio,
                        anm=anm, fs=fs, f0_info=f0_info)
            fn2 = f'{narr}{ti:03d}_{f0}f0_peaky_regress.hdf5'
            write_hdf5(f'{src}/stimuli/speech_regress/{narr}_{f0}f0/{fn2}',
                       data=data, overwrite=True)

# %% create anm regressor for clicks
data = read_hdf5(f'{src}/stimuli/{fn_clicks}')
rates = data['rates']
audio = data['x_10s']
pulses = data['x_pulse_10s']
n_trials = audio.shape[1]
n_ears = audio.shape[2]
n_samples = audio.shape[-1]

pinds = []
anm = np.zeros((len(rates), n_trials, 2, n_ears, n_samples))  # pos, neg
for fi, [f0, rate] in enumerate(zip(f0s, rates)):
    pinds_f0 = []
    for i in range(n_trials):
        inds = []
        for ei in range(n_ears):
            inds.append(np.where(np.abs(pulses[fi, i, ei]) > 0.5)[0])
            anm[fi, i, 0, ei] = amp_models.anm(
                audio[fi, i, ei], fs_audio, stim_db)
            anm[fi, i, 1, ei] = amp_models.anm(
                -audio[fi, i, ei], fs_audio, stim_db)
        pinds_f0.append(inds)
    pinds.append(pinds_f0)
data = dict(
    x=audio,
    pinds=pinds,
    anm=anm,
    rates=rates)

write_hdf5(f'{src}/stimuli/clicks_f0rates_regress.hdf5', data=data,
           overwrite=True)

# %% compile speech
for narr in narrators:
    for f0 in f0s:
        pinds = []
        anm = []
        x = []
        f0_info = []
        for ti in range(n_trials_speech):
            fn = f'{narr}_{f0}f0/{narr}{ti:03d}_{f0}f0_peaky_regress.hdf5'
            data = read_hdf5(f'{src}/stimuli/speech_regress/{fn}')
            pinds.append(data['pinds'])
            anm.append(data['anm'])
            x.append(data['x'])
            f0_info.append(data['f0_info'])
        anm = np.array(anm)
        x = np.array(x)
        f0_info = np.array(f0_info)
        data = dict(x=x, pinds=pinds, anm=anm, f0_info=f0_info, fs=fs_audio)
        write_hdf5(f'{src}/stimuli/{narr}_{f0}f0_regress.hdf5', data=data,
                   overwrite=True)
