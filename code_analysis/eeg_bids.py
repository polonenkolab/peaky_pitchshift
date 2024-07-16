# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:42:15 2024

@author: mpolonen
"""

import numpy as np
import os.path as op
from numpy.testing import assert_array_equal
import mne
from mne_bids.copyfiles import copyfile_brainvision
# from mne_bids import write_raw_bids, make_bids_basename
# from mne_bids.utils import print_dir_tree
from mne_bids import BIDSPath, write_raw_bids, print_dir_tree
from mne_bids.stats import count_events
from bids_validator import BIDSValidator
from expyfun import binary_to_decimals
from expyfun.io import read_hdf5, write_hdf5
import pandas as pd
import json
import uuid
import os
from glob import glob
import pathlib

src = pathlib.Path(__file__).parents[0].resolve().as_posix()
path_bids = f'{src}/data_bids/'
if not os.path.exists(path_bids):
    os.mkdir(path_bids)
# %% parameters
exp = 'peaky_pitch'
task = 'peakypitch'
fs = 10e3

stimuli = ['clicks', 'male', 'female']
f0s = ['low', 'mid', 'high']
ears = ['left', 'right']
narrators = ['male', 'female']
# rates = read_hdf5(stim_path + click_fn)['rates']

idx_stim = {i: si for i, si in enumerate(stimuli)}
idx_f0s = {i: fi for i, fi in enumerate(f0s)}

conditions = [f'{s}_{f}' for s in stimuli for f in f0s]
idx_conds = {c: i for i, c in enumerate(conditions)}

n_stim = len(stimuli)
n_f0s = len(f0s)
n_ears = len(ears)
n_narrs = len(narrators)

dur_stim = 10
dur_stamp = 9.98
n_min_speech = 20
n_min_clicks = 5

n_trials_speech = int(n_min_speech * 60 / dur_stim)
n_trials_clicks = int(n_min_clicks * 60 / dur_stim)

n_subs = 15

n_bits_stim = int(np.ceil(np.log2(n_stim)))
n_bits_f0 = int(np.ceil(np.log2(n_f0s)))
n_bits_trial = int(np.ceil(np.log2(np.max(
    [n_trials_speech, n_trials_clicks]))))
n_bits_stamp = n_bits_stim + n_bits_f0 + n_bits_trial

event_id = {'New Segment/': 0, 'Stimulus/S  1': 1, 'Stimulus/S  2': 2,
            'Stimulus/S  4': 4, 'Stimulus/S  8': 8}

# %% create eeg-bids format
for si in range(n_subs):
    sub = si + 1
    print(f'sub {sub:02d}')
    fn = f'{src}/data_raw/{exp}_{sub:03d}.vhdr'
    raw = mne.io.read_raw_brainvision(fn, preload=False)
    raw.info['line_freq'] = 60

    bids_path = BIDSPath(subject=f'{sub:02d}', task=task, root=path_bids,
                         datatype='eeg')

    # convert to eeg-bids format
    events = mne.events_from_annotations(raw, event_id)[0]

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
    clock_diff = dur_stamp * fs - len_trial
    clock_adjust = (dur_stamp * fs / len_trial).mean()
    fs_adjust = clock_adjust * fs

    start_inds = np.array(np.where(events[:, -1] == 1))[0]
    bits_epoch = [events[start + 1:start + n_bits_stamp + 1, -1] >> 3
                  for start in start_inds]

    stim = np.array([binary_to_decimals(be[
        :n_bits_stim], n_bits_stim)[0] for be in bits_epoch])
    f0 = np.array([binary_to_decimals(be[
        n_bits_stim:n_bits_stim + n_bits_f0], n_bits_f0)[0]
        for be in bits_epoch])
    trial = np.array([binary_to_decimals(be[
        -n_bits_trial:], n_bits_trial)[0] for be in bits_epoch])
    assert len(stim) == len(f0) == len(trial) == len(starts)

    stim_txt = [idx_stim[i] for i in stim]
    f0_txt = [idx_f0s[i] for i in f0]
    cond_txt = [f'{s}_{f}' for s, f in zip(stim_txt, f0_txt)]
    cond = [idx_conds[i] for i in cond_txt]

    stim_file = [f'{ci}/{ci}_{ti:03d}_regress.hdf5'
                 for ci, ti in zip(cond_txt, trial)]

    stim_events = starts.copy()
    stim_events[:, -1] = cond
    stim_events[:, -2] = dur_stim

    write_raw_bids(raw, bids_path, events_data=stim_events,
                   event_id=idx_conds, overwrite=True)

    # modify events file
    eeg_tsv = f'{path_bids}sub-{sub:02d}/eeg/{bids_path.basename}_events.tsv'
    event_file = pd.read_csv(eeg_tsv, sep='\t')
    event_file['duration'] = dur_stim
    event_file['token'] = trial
    event_file['EEG_trigger_start'] = 1
    event_file['EEG_trigger_end'] = 2
    event_file['stim_file'] = stim_file
    with open(eeg_tsv, 'w') as write_tsv:
        write_tsv.write(event_file.to_csv(sep='\t', index=False))

    # modify json file
    eeg_json = f'{path_bids}sub-{sub:02d}/eeg/{bids_path.basename}_eeg.json'
    with open(eeg_json, 'r') as f:
        json_file = json.load(f)
        json_file['PowerLineFrequency'] = 60
        json_file['EEGReference'] = 'placed on FCz'
        json_file['EEGGround'] = 'placed on Fpz'
        json_file['ManufacturersModelName'] = 'actiCHamp with EP-PreAmp'
        json_file['SoftwareVersions'] = 'PyCorder V1.0.9'
    with open(eeg_json, 'w') as f:
        json.dump(json_file, f, indent=4)

# assert False
# %% subjects tsv and json
subjects_tsv = f'{path_bids}participants.tsv'
subjects_json = f'{path_bids}participants.json'

subjects_data = pd.read_csv(f'{src}/{exp}_participant_info.csv', sep=',')
with open(subjects_tsv, 'w') as write_tsv:
    write_tsv.write(subjects_data.to_csv(sep='\t', index=False))

freqs = [500, 1000, 2000, 4000, 8000]
ears = ['left', 'right']
hl_descriptions = []
for ei in ears:
    for fi in freqs:
        hl_descriptions += [{
            'Description': f'hearing threshold for {fi} Hz in the {ei} ear',
            'Units': 'dB HL'}]
other_descriptions = []
for di in ["Participant",
           "Age",
           "Gender",
           "Race",
           "Ethnicity"]:
    other_descriptions += [{'Description': di}]

subject_descriptions = other_descriptions + hl_descriptions + [
    {'Description': 'notes about EEG files and trial number repeats'}]

s_info = {}
for item, descr in zip(subjects_data.columns, subject_descriptions):
    s_info[item] = descr
s_info['age']['Units'] = 'years'

with open(subjects_json, 'w') as f:
    json.dump(s_info, f, indent=4)

# %% overall _events.json and _eeg.json

task_eeg_json = f'{path_bids}task-{task}_eeg.json'
json_file_overall = json_file
del json_file_overall['RecordingDuration']
with open(task_eeg_json, 'w') as f:
    json.dump(json_file_overall, f, indent=4)

event_descriptions = [
    "Onset of the audio event.",
    "Duration of the audio event.",
    "Which audio type was heard by the subject in this trial.",
    '''The code associated with a trial type. NOTE: this is NOT the trigger
    type in EEG annotations but these are separated by 2 triggers in EEG''',
    "The sample within the EEG data at which an event occurred.",
    '''The file number of the trial type. E.g., for male_high trial 1 you would 
        load 'male001_highf0'.''',
    "The trigger values in the EEG annotations for the event onset.",
    "The trigger values in the EEG annotations for 20 ms before event offset.",
    "The file loaded to play the audio (called 'x' in the hdf5 dictionary)"]

task_event_json = f'{path_bids}task-{task}_events.json'
event_file_overall = {}
for item, descr in zip(event_file.columns, event_descriptions):
    event_file_overall[item] = {'Description': descr}
event_file_overall['onset']['Units'] = 's'
event_file_overall['duration']['Units'] = 's'

with open(task_event_json, 'w') as f:
    json.dump(event_file_overall, f, indent=4)

# %% README

txt = """
README
------
Details related to access to the data
-------------------------------------
Please contact the following authors for further information:
    Melissa Polonenko(email: mpolonen@umn.edu)
    Ross Maddox (email: rkmaddox@med.umich.edu)

Overview
--------
This is the "peaky_pitchshift"" dataset for the paper
Polonenko MJ & Maddox RK (2024), with citation listed below.

BioRxiv: 

Auditory brainstem responses (ABRs) were derived to continuous peaky speech
from two talkers with different fundamental frequencies (f0s) and from clicks
that have mean stimulus rates set to the mean f0s. Data was collected from
May to June 2021.

Aims:
    1) replicate the male/female talker effect with each at their natural f0
    2) systematically determine if f0 is the main driver of this talker difference
    3) evaluate if the f0 effect resembles the click rate effect

The details of the experiment can be found at Polonenko & Maddox (2024). 

Stimuli:
    1) randomized click trains at 3 stimulus rates (123, 150, 183 Hz), 
    30 x 10 s trials each for a total of 90 trials (15 min, 5 min each rate)
    2) peaky speech for a male and female narrator at 3 f0s (123, 150, 183 Hz),
    120 x 10 s trials each of the 6 narrator-f0 combo for a total of 720 trials
    (2 hours, 20 min each)
    
    NOTE: f0s used: original f0s (low & high respectively) and f0s
    shifted to the other narrator's f0 and an f0 at the midpoint between the f0s.
    click rates used: set to the mean f0s used for the speech

The code for stimulus preprocessing and EEG analysis is available on Github:
    https://github.com/polonenkolab/peaky_pitchshift

Format
------
The dataset is formatted according to the EEG Brain Imaging Data Structure. It 
includes EEG recording from participant 01 to 15 in raw brainvision format
(3 files: .eeg, .vhdr, .vmrk) and stimuli files in format of .hdf5. The stimuli
files contain the audio ('x'), and regressors for the deconvolution 
('pinds' are the pulse indices, 'anm' is an auditory nerve model regressor,
 which was used during analyses but was not included as part of the article). 

Generally, you can find detailed event data in the .tsv files and descriptions
in the accompanying .json files. Raw eeg files are provided in the Brain
Products format.

Participants
------------
15 participants, mean ± SD age of 24.1 ± 6.1 years (19-35 years)

Inclusion criteria:
    1) Age between 18-40 years
    2) Normal hearing: audiometric thresholds 20 dB HL or better from 500 to 8000 Hz
    3) Speak English as their primary language

Please see participants.tsv for more information.

Apparatus
---------
Participants sat in a darkened sound-isolating booth and rested or watched
silent videos with closed captioning. Stimuli were presented at an average level
of 65 dB SPL and a sampling rate of 48 kHz through ER-2 insert earphones
plugged into an RME Babyface Pro digital sound card. Custom python scripts
using expyfun were used to control the experiment and stimulus presentation.

Details about the experiment
----------------------------
For a detailed description of the task, see Polonenko & Maddox (2024) and the
supplied `task-peaky_pitch_eeg.json` file. The 6 peaky speech conditions
(2 narrators x 3 f0s) were randomly interleaved for each block of trials
(i.e., for trial 1, the 6 conditions were randomized) and the story token
was randomized. This means that the participant would not be able to follow
the story. For clicks the trials were not randomized (already random clicks).

Trigger onset times in the tsv files have already been corrected for the tubing
delay of the insert earphones (but not in the events of the raw files).
Triggers with values of "1" were recorded to the onset of the 10 s audio, and
shortly after triggers with values of "4" or "8" were stamped to indicate the
overall trial number out of 120 for each speech conditon and out of 30 for each
click condition. This was done by converting the decimal trial number to bits,
denoted b, then calculating 2 ** (b + 2). We've specified these trial numbers
and more metadata of the events in each of the '*_eeg_events.tsv" file, which
is sufficient to know which trial corresponded to which type of stimulus
(clicks, male narrator, female narrator), which f0 (low, mid, high), and which
file - e.g., male_low_000_regress.hdf5 for the male narrator with the low f0.

"""

README = f'{path_bids}README'
with open(README, "w", encoding="utf-8") as fout:
    print(txt, file=fout)

# %%
# check and compare with standard
print_dir_tree(path_bids)

# %%
# bids_path
files = sorted(glob(f'{path_bids}*'))
file = []
[file.append(f) for f in files if f.split('/')[-1][:3] != 'sub']
for fn in file:
    print(fn[len(path_bids) - 1:])
    print(BIDSValidator().is_bids(fn[len(path_bids) - 1:]))
for si in range(n_subs):
    spath = f'{path_bids}sub-{si+1:02d}/'
    print(f'\n== {spath} ==')
    file = sorted(glob(f'{spath}_*'))
    for fn in file:
        print(fn.split('/')[-1])
        print(BIDSValidator().is_bids(fn[len(path_bids) - 1:]))
    trues = []
    files = sorted(glob(spath + 'eeg/' + spath.split('/')[-2] + '_*'))
    for fn in files:
        # print(fn.split('/')[-1])
        # print(BIDSValidator().is_bids(fn[len(bids_path) - 1:]))
        if BIDSValidator().is_bids(fn[len(path_bids) - 1:]):
            trues += [fn]
    if len(trues) == len(files):
        print('True')
    else:
        print('False')
