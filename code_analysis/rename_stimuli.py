# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:00:48 2024

@author: mpolonen
"""

import numpy as np
from glob import glob
import os
import shutil
from expyfun.io import read_hdf5, write_hdf5


# %% speech
path_in = 'C:/code/peaky_pitch/stimuli/speech_regress/'
path_out = 'C:/code/peaky_pitch/stimuli/stimuli_bids/'

narrators = ['male', 'female']
f0s = ['low', 'mid', 'high']

for narr in narrators:
    for f0 in f0s:
        print(f'copying {narr}_{f0}')
        files = sorted(glob(f'{path_in}{narr}_{f0}/*.hdf5'))
        if not os.path.exists(f'{path_out}{narr}_{f0}/'):
            os.mkdir(f'{path_out}{narr}_{f0}/')
        for name_old in files:
            token = int(name_old.split('_')[-4][-3:])
            name_new = f'{path_out}{narr}_{f0}/{narr}_{f0}_{token:03d}_regress.hdf5'
            shutil.copy(name_old, name_new)

# %% clicks
data = read_hdf5('C:/code/peaky_pitch/stimuli/clicks_f0rates_regress.hdf5')
rates = data['rates']
n_tokens = data['x'].shape[1]
dur_stim = 10
# x: 3, 60, 2, 480k
# anm: 3, 60, 2, 2, 480k
# pinds: 3, 60, 2, var

for fi, [rate, f0] in enumerate(zip(rates, f0s)):
    print(f'saving clicks {f0}')
    p_out = f'{path_out}clicks_{f0}/'
    if not os.path.exists(p_out):
        os.mkdir(p_out)
    for i in range(n_tokens):
        fs = int(data['x'].shape[-1] / dur_stim)
        anm = data['anm'][fi, i]
        x = data['x'][fi, i]
        pinds = data['pinds'][fi][i]
        data_save = dict(fs=fs, rate=rate, anm=anm, x=x, pinds=pinds)
        write_hdf5(f'{p_out}clicks_{f0}_{i:03d}_regress.hdf5', data_save)
