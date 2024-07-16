#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 01:12:51 2021

@author: mpolonenko
"""
import numpy as np
import os
from expyfun.io import read_wav, write_wav, write_hdf5, read_hdf5
from glob import glob
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
import json
import math
plt.ion()

# %% User settings

stories = ['alchemyst', 'wrinkle']
narrators = ['male', 'female']
opath = '/mnt/data/peaky_pitch/stimuli/'
wav_in_path = '/mnt/data2/pip_speech/audio_books_48k/{}/'
wav_out_path = opath + 'slices/stories/'

concat_root = '{}_{:02}files'  # story, n_files
shift_root = '{}_shift{:1}.wav'

factors = [2, 1 / 2]  # shift male up and female down
conditions = np.array([0, 0.5, 1])  # no shift, half-way, full shift
f0s = ['low', 'mid', 'high']

stim_dur = 64.
stim_fs = int(48e3)

f0_min = [60, 90]
f0_max = [350, 500]

crop_sec_beg = 3
crop_sec_end = 1
stim_dur = 64
n_min_need = 20

crop_dur = stim_dur - crop_sec_beg - crop_sec_end
n_files = np.ceil(n_min_need * 60 / crop_dur).astype(int)

n_crop_beg = int(crop_sec_beg * stim_fs)
n_crop_end = int(crop_sec_end * stim_fs)

dur_fade = .03
dur_slice = 10

n_stories = len(stories)

# %% define functions


def checkpath(path):  # helper function so saving figs is easier
    if not os.path.isdir(path):
        os.makedirs(path)


def shift_pitch(sound, hz, fmin, fmax):
    manipulation = call(sound, "To Manipulation", 0.01, fmin, fmax)
    type(manipulation)
    manipulation.class_name
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Shift frequencies", sound.xmin, sound.xmax, hz, 'Hertz')
    call([pitch_tier, manipulation], "Replace pitch tier")
    return call(manipulation, "Get resynthesis (overlap-add)")


def change_pitch(sound, factor, fmin, fmax):
    manipulation = call(sound, "To Manipulation", 0.01, fmin, fmax)
    type(manipulation)
    manipulation.class_name
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
    call([pitch_tier, manipulation], "Replace pitch tier")
    return call(manipulation, "Get resynthesis (overlap-add)")


def make_slices(x, fs_x, fade_dur, n_slices, slice_dur, save_wav=False):
    x_all = []
    temp = 0
    for ni in range(n_slices):
        x_slice = x[temp:temp + int(slice_dur * fs_x)]
        t_in = np.arange(int(fade_dur * fs_x))
        t_out = np.arange(int(len(x_slice) - fade_dur * fs_x), len(x_slice), 1)
        x_slice[t_in] = x_slice[t_in] * [
            math.sin(2 * math.pi / (fade_dur * 4 * fs_x) * i) for i in t_in]
        x_slice[t_out] = x_slice[t_out] * [
            math.cos(2 * math.pi / (fade_dur * 4 * fs_x) * i) for i in t_out]
        x_slice *= 0.01 / x_slice.std()
        temp += int(slice_dur * fs_x)
        x_all.append(x_slice)
        if save_wav:
            write_wav(wav_out_path + story + '{:04}.wav'.format(ni), x_slice,
                      fs_x, overwrite=True)
    return x_all


# %% combine and crop out repeated info and 1s fade in/out
f0_info = np.zeros((n_stories, 4))
checkpath(wav_out_path)
for st, story in enumerate(stories):
    print('\n== crop & combine {} files ==\n'.format(story))
    files = sorted(glob(wav_in_path.format(story) + '*.wav'))[:n_files]
    wavs = np.zeros((n_files, int(stim_dur * stim_fs)))
    for fi, fn in enumerate(files):
        wavs[fi], fs = read_wav(fn)
    wavs = wavs[:, n_crop_beg:-n_crop_end]
    fn = wav_out_path + concat_root.format(story, n_files)
    write_wav(fn + '.wav', np.concatenate(wavs), stim_fs, overwrite=True)
    parselmouth.praat.run_file(opath + 'wav_to_pitch.praat', fn,
                               str(f0_min[st]), str(f0_max[st]))
    with open(fn + '.Pitch') as fid:
        pitch = list(np.copy(fid.readlines()))[0][:-1]
    fmin, fmax, fmean, fstd = [float(d) for d in pitch.split('_')]
    f0_info[st] = np.array([fmin, fmax, fmean, fstd])

# %% determine shift semitones
print('determining semitone shifts')
f0_diff = np.diff(f0_info[..., -2])[0]
n_semitones_diff = 12 * np.log2(f0_info[..., -2][1] / f0_info[..., -2][0])
semitone_shifts = conditions * n_semitones_diff
shifts = semitone_shifts / 12
n_shifts = len(shifts)

f0_changed = np.zeros((n_stories, n_shifts))
for fi, [f0, factor] in enumerate(zip(f0_info[..., -2], factors)):
    for sh, shift in enumerate(shifts):
        if fi == 0:
            f0_changed[fi, sh] = f0 * factor ** shift
        else:
            f0_changed[fi, abs(sh - n_shifts + 1)] = f0 * factor ** shift
diffs = np.diff(f0_changed, axis=0)
print(np.round(diffs))

# %% shift files
wavs = np.zeros((len(stories), len(shifts), int(crop_dur * stim_fs * n_files)))
f0_info_shifted = np.zeros((n_stories, n_shifts, 4))
for st, [story, factor] in enumerate(zip(stories, factors)):
    print('\n== shifting {} =='.format(story))
    fn = wav_out_path + concat_root.format(story, n_files) + '.wav'
    for sh, shift in enumerate(shifts):
        print('shift {:.1f} semitones'.format(semitone_shifts[sh]))
        sound = parselmouth.Sound(fn)
        sound_sh = change_pitch(sound, factor ** shift, f0_min[st], f0_max[st])
        type(sound_sh)
        wavs[st, sh] = sound_sh.values
        fn2 = wav_out_path + '{}_shift{}'.format(story, sh)
        write_wav(fn2 + '.wav', wavs[st, sh], stim_fs, overwrite=True)
        parselmouth.praat.run_file(opath + 'wav_to_pitch.praat', fn2,
                                   str(f0_min[0]), str(f0_max[1]))
        with open(fn2 + '.Pitch') as fid:
            pitch = list(np.copy(fid.readlines()))[0][:-1]
        fmin, fmax, fmean, fstd = [float(d) for d in pitch.split('_')]
        f0_info_shifted[st, sh] = np.array([fmin, fmax, fmean, fstd])

# save info
for st, [story, data] in enumerate(zip(stories, f0_info_shifted)):
    with open(wav_out_path + 'pitchinfo_{}.txt'.format(story), 'w') as outfile:
        json.dump(dict(f0_min=list(data[:, 0]), f0_max=list(data[:, 1]),
                       f0_mean=list(data[:, 2]), f0_std=list(data[:, 3]),
                       semitone_shifts=list(semitone_shifts)), outfile)

# %% sort by low, mid, high f0 instead of shift
f0_sorted = np.array([f0_info_shifted[0], f0_info_shifted[1, ::-1]])
wavs_sorted = np.array([wavs[0], wavs[1][::-1]])

print('Differences for low, mid, high f0:')
for i, f in enumerate(['min', 'max', 'mean', 'std']):
    print('{}: {}'.format(f, np.diff(f0_sorted, axis=0)[0][:, i]))

# %% make slices
n_slices_need = int(np.ceil(wavs_sorted.shape[-1] / dur_slice / stim_fs))
assert(wavs_sorted.shape[-1] % (dur_slice * stim_fs) == 0)

wavs_sliced = np.zeros((n_stories, n_shifts, n_slices_need,
                        int(dur_slice * stim_fs)))
for st, story in enumerate(stories):
    for sh, shift in enumerate(shifts):
        print('\n== slicing {}, shift {} =='.format(story, sh))
        wavs_sliced[st, sh] = make_slices(wavs_sorted[st, sh], stim_fs,
                                          dur_fade, n_slices_need, dur_slice)

# %% save
write_hdf5(wav_out_path + 'pitch_shift_data.hdf5', dict(
    n_semitones_diff=n_semitones_diff,
    semitone_shifts=semitone_shifts,
    f0_sorted=f0_sorted,
    wavs=wavs_sorted,
    wavs_sliced=wavs_sliced,
    fs=stim_fs
    ), overwrite=True)
