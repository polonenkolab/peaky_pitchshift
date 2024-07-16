#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:56:34 2021

@author: mpolonenko
"""
import numpy as np
# import matplotlib.pyplot as plt
from expyfun.io import read_hdf5, write_hdf5
import pip_trains_rme
import os

# %%
opath = '/mnt/data/peaky_pitch/'
stim_path = opath + 'stimuli/slices/clicks/'
click_root = 'clicks_{}.hdf5'

if not os.path.exists(stim_path):
    os.makedirs(stim_path)

# %%
f0info = read_hdf5(opath + 'stimuli/slices/stories/pitch_shift_data.hdf5')[
    'f0_sorted']
rates = np.round(f0info.mean(0)[..., -2]).astype(int)

n_min_need = 10
n_min = n_min_need / 2  # b/c interleave n_tokens, so doubles
for rate in rates:
    pip_trains_rme.make_pabr_stim(stim_path, rate=rate, n_minutes=n_min,
                                  do_pips=False)

# %%
x_all = []
x_pulse = []

for rate in rates:
    data = read_hdf5(stim_path + click_root.format(rate))
    x_all += [data['x']]
    x_pulse += [data['x_pulse']]

x_all = np.array(x_all)
x_pulse = np.array(x_pulse)

if x_all.shape[-2] == x_pulse.shape[-2] == 1:
    x_all = x_all.mean(-2)
    x_pulse = x_pulse.mean(-2)

# %%
write_hdf5(stim_path + 'clicks_f0rates.hdf5', dict(
    x=x_all,
    x_pulse=x_pulse,
    rates=rates,
    n_tokens=x_all.shape[1],
    n_ears=x_all.shape[-2],
    fs=x_all.shape[-1]), overwrite=True)
