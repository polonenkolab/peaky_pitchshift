#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:45:50 2021

@author: mpolonenko
"""
import numpy as np
from expyfun import ExperimentController, decimals_to_binary
from expyfun.visual import ProgressBar
from expyfun.stimuli import TrackerDealer, TrackerUD
from expyfun.io import read_wav, read_hdf5
from glob import glob
import io
import json
import hashlib
import shutil
import time
import sys
import matplotlib.pyplot as plt
plt.ion()
# %% parameters

opath = 'C:/Users/Maddox/Code/Experiments/pip_speech_shift/'
file_root = 'stimuli/{}_{}f0/{}{:03}_{}f0_broadband.wav'
click_root = 'stimuli/clicks_f0rates.hdf5'
dpath = opath + 'data/'

# hardware
fs_stim = 48000
n_channels = 2
rms = 0.01
stim_db = 65
stim_dur = 10.
click_dur = 10.
ac = dict(TYPE='sound_card', SOUND_CARD_BACKEND='rtmixer',
          SOUND_CARD_FIXED_DELAY=0.000,
          SOUND_CARD_TRIGGER_CHANNELS=2,
          SOUND_CARD_TRIGGER_ID_AFTER_ONSET=True,
          SOUND_CARD_DRIFT_TRIGGER='end'
          )
# general experiment
pause_dur = 0.03
narrators = ['male', 'female']
f0s = ['low', 'mid', 'high']
stimuli = ['speech', 'clicks']
stim_index = {st: i for i, st in enumerate(stimuli)}
pause_dur_click = 0.03

n_stim = len(stimuli)
n_narrs = len(narrators)
n_f0s = len(f0s)
n_min_cond = 20
n_trials = int(n_min_cond * 60 / stim_dur)
n_trials_total = n_narrs * n_f0s * n_trials
n_min_clicks = 5.
n_trials_clicks = int(n_min_clicks * 60 / click_dur)
n_trials_clicks_total = n_trials_clicks * n_f0s

n_bits_stim = int(np.ceil(np.log2(n_stim)))
n_bits_narr = int(np.ceil(np.log2(n_narrs)))
n_bits_f0 = int(np.ceil(np.log2(n_f0s)))
n_bits_trial = int(np.ceil(np.log2(n_trials)))
n_bits_click = int(np.ceil(np.log2(n_trials_clicks)))

trial_order = np.zeros((n_narrs, n_f0s, n_trials))
for ni in range(n_narrs):
    for fi in range(n_f0s):
        trial_order[ni, fi] = np.random.permutation(n_trials)
trial_order = trial_order.astype(int)

# tracker params
up = 1
down = 1
step_size_up = 0
step_size_down = step_size_up
stop_trials = n_trials
stop_reversals = np.inf
start_value = stim_db
change_indices = None
change_rule = 'trials'
x_min = stim_db
x_max = stim_db
max_lag = 1
pace_rule = 'trials'
rand_dealer = None  # np.random.RandomState(2),random seed to select trial type

click_stop_trials = n_trials_clicks

instructions_clicks = '''You will passively listen to {} minutes of clicks.
                      Press '1' to start the experiment.'''
instructions_speech = '''You will passively listen to {} minutes of speech.
There are 10 s segments of speech that are randomly chosen from two audio
books. Press '1' to start the experiment.'''

# %%
rme_setting = input('Is RME buffer set to 1024 samples? [y]: ')
if not rme_setting == '':
    assert(False)
# click setting
if_click = input('Do you want to run click session [y]: ')
if if_click == '':
    do_click = True
else:
    do_click = False

# %%  import stimuli
wavs = np.zeros((n_narrs, n_f0s, n_trials, int(stim_dur * fs_stim)))
wavs.fill(np.nan)
for ni, narr in enumerate(narrators):
    for fi, f0 in enumerate(f0s):
        for ti in range(n_trials):
            wavs[ni, fi, ti], fs = read_wav(opath + file_root.format(
                narr, f0, narr, ti, f0))
            assert fs == fs_stim

if do_click:
    clicks = read_hdf5(opath + click_root)['x']
    rates = read_hdf5(opath + click_root)['rates']

# %% set high priorty


def setpriority(pid=None, priority=1):
    """ Set The Priority of a Windows Process.  Priority is a value between
        0-5 where 2 is normal priority.  Default sets the priority of the
        current python process but can take any valid process ID. """
    import win32api
    import win32process
    import win32con
    priorityclasses = [win32process.IDLE_PRIORITY_CLASS,
                       win32process.BELOW_NORMAL_PRIORITY_CLASS,
                       win32process.NORMAL_PRIORITY_CLASS,
                       win32process.ABOVE_NORMAL_PRIORITY_CLASS,
                       win32process.HIGH_PRIORITY_CLASS,
                       win32process.REALTIME_PRIORITY_CLASS]
    if pid is None:
        pid = win32api.GetCurrentProcessId()
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
    win32process.SetPriorityClass(handle, priorityclasses[priority])


setpriority(priority=4)
status_string = ''

# %%
with ExperimentController('peaky_pitch',
                          audio_controller=ac,
                          response_device='keyboard',
                          stim_rms=rms,
                          stim_fs=fs_stim,
                          stim_db=stim_db,
                          noise_db=-np.inf,
                          output_dir='data',
                          window_size=[1200, 900],
                          screen_num=0,
                          full_screen=True,
                          force_quit=['end'],
                          trigger_controller='sound_card',
                          session='test',
                          check_rms=None,
                          suppress_resamp=True,
                          version='dev',
                          n_channels=n_channels,
                          verbose=True) as ec:
    ec.set_visible(True)
    ec.write_data_line('n_bits_stim', n_bits_stim)
    ec.write_data_line('n_bits_narrator', n_bits_narr)
    ec.write_data_line('n_bits_f0', n_bits_f0)
    ec.write_data_line('n_bits_trial', n_bits_trial)
    ec.write_data_line('n_bits_click', n_bits_click)

    if do_click:
        pb_tot = ProgressBar(ec, [0, -0.5, 1.5, 0.2], colors=['b', 'w'])
        ec.screen_prompt(instructions_clicks.format(int(n_min_clicks * n_f0s)),
                         live_keys=['1'], wrap=True)
        start_times = [ec.current_time]
        tr_num = 0
        tr_ud_click = [TrackerUD(
            ec, up, down, step_size_up, step_size_down, stop_reversals,
            click_stop_trials, start_value, change_indices, change_rule,
            x_min, x_max) for _ in range(n_f0s)]
        td_click = TrackerDealer(ec, tr_ud_click, max_lag, pace_rule,
                                 rand_dealer)
        ec.listen_presses()
        while not td_click.stopped:
            for ss, level in td_click:
                # Press 1 to pause
                if len(ec.get_presses(live_keys=['1'], timestamp=False)):
                    ec.write_data_line('pause', 'start')
                    ec.screen_prompt("Press '1' to continue.", live_keys=['1'],
                                     wrap=False)
                    ec.write_data_line('pause', 'stop')
                    ec.listen_presses()
                ti = td_click.trackers[ss].n_trials
                fi = ss[0]
                si = stim_index['clicks']
                assert(clicks[ss][ti].shape[0] == 2)
                trial_audio = clicks[ss][ti]
                ec.identify_trial(
                    ec_id='{}_{}Hz_{:03}'.format(stimuli[si], rates[fi], ti),
                    ttl_id=decimals_to_binary(
                        [si, fi, ti], [n_bits_stim, n_bits_f0, n_bits_trial]))
                ec.load_buffer(trial_audio)
                # set display
                ttr = (click_dur + pause_dur_click) * (n_trials_clicks_total -
                                                       tr_num)
                status_string = (
                    '\nPlaying trial {} / {}\n'.format(tr_num + 1,
                                                       n_trials_clicks_total) +
                    '\nTotal time remaining: {} min\n'.format(int(ttr // 60)) +
                    '\nFinish time: {}\n'.format(time.strftime(
                        '%I:%M', time.localtime(time.time() + ttr))))
                ec.screen_text(status_string + "\nPress '1' to pause.",
                               wrap=True)
                pb_tot.update_bar((100 * (tr_num + 1)) / n_trials_clicks_total)
                pb_tot.draw()
                ec.flip()
                # play the file
                start_times += [ec.current_time]
                ec.start_stimulus(flip=False)
                ec.wait_secs(trial_audio.shape[-1] / fs_stim + pause_dur_click)
                td_click.respond(True)
                ec.stop()
                ec.trial_ok()
                ec.write_data_line('epoch number', tr_num)
                tr_num += 1
                ec.check_force_quit()

    pb_tot = ProgressBar(ec, [0, -0.5, 1.5, 0.2], colors=['b', 'w'])
    ec.screen_prompt(instructions_speech.format(int(
            n_min_cond * n_f0s * n_narrs)), live_keys=['1'], wrap=True)
    start_times = [ec.current_time]
    tr_num = 0
    tr_ud_track = [[TrackerUD(
        ec, up, down, step_size_up, step_size_down, stop_reversals,
        stop_trials, start_value, change_indices, change_rule, x_min, x_max)
        for _ in range(n_f0s)] for _ in range(n_narrs)]
    td_track = TrackerDealer(ec, tr_ud_track, max_lag, pace_rule,
                             rand_dealer)
    ec.listen_presses()
    while not td_track.stopped:
        # retrieve audio
        for ss, level in td_track:
            # Press 1 to pause
            if len(ec.get_presses(live_keys=['1'], timestamp=False)):
                ec.write_data_line('pause', 'start')
                ec.screen_prompt("Press '1' to continue.", live_keys=['1'],
                                 wrap=False)
                ec.write_data_line('pause', 'stop')
                ec.listen_presses()
            tr = td_track.trackers[ss].n_trials
            ti = trial_order[ss][tr]
            ni = ss[0]
            fi = ss[1]
            si = stim_index['speech']
            trial_audio = np.tile(wavs[ss][ti], [2, 1])
            ec.identify_trial(
                ec_id='{}_{}_{}_{:03}'.format(stimuli[si], narrators[ni],
                                              f0s[fi], ti),
                ttl_id=decimals_to_binary(
                    [si, ni, fi, ti], [n_bits_stim, n_bits_narr, n_bits_f0,
                                       n_bits_trial]))
            ec.load_buffer(trial_audio)
            # set display
            ttr = (stim_dur + pause_dur) * (n_trials_total - tr_num)
            status_string = (
                '\nPlaying trial {} / {}\n'.format(tr_num + 1, n_trials_total)
                + '\nTotal time remaining: {} min\n'.format(int(ttr // 60)) +
                '\nFinish time: {}\n'.format(time.strftime(
                    '%I:%M', time.localtime(time.time() + ttr))))
            ec.screen_text(status_string + "\nPress '1' to pause.", wrap=True)
            pb_tot.update_bar((100. * (tr_num + 1)) / n_trials_total)
            pb_tot.draw()
            ec.flip()
            # play the file
            start_times += [ec.current_time]
            ec.start_stimulus(flip=False)
            ec.wait_secs(trial_audio.shape[-1] / fs_stim + pause_dur)
            td_track.respond(True)
            ec.stop()
            ec.trial_ok()
            ec.write_data_line('epoch number', tr_num)
            tr_num += 1
            ec.check_force_quit()
    if tr_num == n_trials_total:
        ec.wait_secs(5)
        ec.set_background_color('b')
        ec.system_beep()
        ec.screen_text('All done!', wrap=False, fone_size=48, color='k')
        ec.flip()
        ec.wait_for_presses(10, live_keys=['1'])
ec.system_beep()
