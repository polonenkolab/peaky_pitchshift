# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:42:38 2024

@author: mpolonen

NOTE: can be used to pick waves I, III, V for both regressors (pulses, anm)
Gives peak and trough amplitudes and latencies, as well as the overall
peak-trough amplitude
Best for plotting chosen peaks on waveforms!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from expyfun.io import write_hdf5, read_hdf5
import scipy.signal as sig
import datetime
import json
import pathlib
import os

# %% user settings
src = pathlib.Path(__file__).parents[0].resolve().as_posix()
exp = 'peakypitch'
n_subs = 15
# input
marker = input('Your initials: ')
sub_start = input(f'Which subject to start with? (total: {n_subs}): ')
sub_end = input(f'Which subject to end with? [{n_subs}]: ')
filtering = input('150-2000 Hz ok? [y or n, default = y]: ')
if sub_start in ['', '1']:
    sub_start = 1
if sub_end in ['', f'{n_subs}']:
    sub_end = n_subs
if filtering in ['y', 'Y', '']:
    l_freq = 150.
    h_freq = 2000.
else:
    l_freq = input('Enter lower cutoff in Hz (i.e., highpass): ')
    l_freq = float(l_freq)
    h_freq = input('Enter higher cutoff in Hz (i.e., lowpass): ')
    h_freq = float(h_freq)

# convert to variables
sub_start = int(sub_start)
sub_end = int(sub_end)
subs_run = np.arange(sub_start, sub_end + 1, 1)
n_subs_run = len(subs_run)
subs = np.arange(1, n_subs + 1)
idx_subs = {si: i for i, si in enumerate(subs)}

# files
if not os.path.exists(f'{src}/peaks/'):
    os.mkdir(f'{src}/peaks/')
if not os.path.exists(f'{src}/peaks/{marker}/'):
    os.mkdir(f'{src}/peaks/{marker}/')
sv_root = f'{src}/peaks/{marker}/{marker}_{exp}_sub' + \
    '{:02d}' + f'_{int(l_freq)}-{int(h_freq)}Hz'
overwrite_file = True
save_fig = True
save_hdf5 = True
save_txt = False

# %% settings
x_min = 0  # ms
x_max = 14  # ms
window = x_max - x_min

lw = [1, 2]  # when not marked/marked (to help see which ones are not done yet)
y_display = 0.1  # how close the waveforms are plotted
picker_size = [2, 6]  # [3, 15]
y_labelshiftprop = 0.15

regressors = ['pulses', 'anmp']
types = ['clicks', 'male', 'female']
f0s = ['low', 'mid', 'high']
peaks = ['I', 'III', 'V']
reps = ['evens', 'odds']

stimuli = [f'{ti}_{ri}' for ri in regressors for ti in types]
stim_shifts = [window * i for i in range(len(stimuli))]
idx_stim = {i: si for i, si in enumerate(stimuli)}

n_stim = len(stimuli)
n_reg = len(regressors)
n_pks = len(peaks)
n_typ = len(types)
n_f0s = len(f0s)
n_reps = len(reps)

start, stop = 0.42, 0.99
colors_black = plt.cm.Greys(np.linspace(start, stop, n_f0s))[::-1]
colors_blue = plt.cm.Blues(np.linspace(start, stop, n_f0s))[::-1]
colors_orange = plt.cm.Oranges(np.linspace(start, stop, n_f0s))[::-1]
colors_stim = np.array([colors_black, colors_blue, colors_orange])

# %% load data
fs = int(10e3)
tmin = -1
tmax = 1
n_pre = int(np.abs(tmin) * fs)
n_post = int(np.abs(tmax) * fs)
t = np.arange(-n_pre, n_post) / float(fs)
t_ms = t * 1e3

# load info
print('loading data')
data = read_hdf5(f'{src}/data_plot/data_eeg_subs.hdf5')
w_all = data['w_all']
w_reps = data['w_reps']
# dims (subs, reg, sitm, f0, reps, samples)

# filter
print('filtering data')
br, ar = sig.butter(1, np.array(
    [l_freq, h_freq]) / (fs / 2.), btype='bandpass')
w_abr = sig.lfilter(br, ar, w_all)
w_rep = sig.lfilter(br, ar, w_reps)

# %% define functions


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def onpick(event):
    global points_clicked
    global latency
    global amplitude
    global lat1
    global lat2
    global amp1
    global amp2
    thisline = event.artist
    label = thisline.get_label()

    xdata, ydata = thisline.get_data()
    if label[:8] != 'landmark':
        ind = event.ind
        points = np.array([xdata[ind], ydata[ind]])
        plt.plot(points[0], points[1], 'k+-', markersize=12, lw=1.5,
                 picker=True, pickradius=picker_size[1],
                 label='landmark' + label)
    else:
        thisline.remove()
        label = label[8:]

    for ch in ax.get_children():
        if len(str(ch.get_label())) >= 8 and str(
                ch.get_label())[:8] == 'landmark':
            ch.set_data((np.array([d[0]]) for d in ch.get_data()))

    points_clicked[label] = np.array([ch.get_data() for
                                      ch in ax.get_children() if
                                      ch.get_label() == 'landmark' + label])[
                                          ..., 0]

    if label[:8] == 'landmark':
        parent_label = label[8:]
    else:
        parent_label = label

    parent_line = ax.get_children()[[a.get_label() for a in
                                     ax.get_children()].index(parent_label)]

    if len(points_clicked[label]) == 2:
        if np.argmin(points_clicked[label][:, 0]) == 1:  # if in wrong order
            points_clicked[label] = points_clicked[label][::-1]  # flip it
        latency[label] = points_clicked[label][0, 0] - x_shift[label]
        amplitude[label] = (points_clicked[label][0, 1] - points_clicked[
            label][1, 1]) / y_zoom[label]
        # amplitude[label] /= y_zoom[label]
        lat1[label] = points_clicked[label][0, 0] - x_shift[label]
        lat2[label] = points_clicked[label][1, 0] - x_shift[label]
        amp1[label] = points_clicked[label][0, 1] / y_zoom[label]
        amp2[label] = points_clicked[label][1, 1] / y_zoom[label]
        parent_line.set_linewidth(lw[1])
        # so can plot later on the waveforms
    else:
        latency[label] = np.nan
        amplitude[label] = np.nan
        lat1[label] = np.nan
        lat2[label] = np.nan
        amp1[label] = np.nan
        amp2[label] = np.nan
        parent_line.set_linewidth(lw[0])
    print(points_clicked[label])


def onclose(event):
    global time_picking
    global end_time
    end_time = datetime.datetime.now()
    time_picking = (end_time - start_time).total_seconds()
    print('Time to pick peaks:' + str(time_picking / 60) + ' min')
    print(latency)
    print(amplitude)
    if save_fig:
        plt.savefig(sv_root.format(sub) + '.png')
    if save_hdf5:
        write_hdf5(sv_root.format(sub) + '.hdf5', dict(
            time_picking=time_picking,
            latency=latency,
            amplitude=amplitude,
            lat1=lat1,
            lat2=lat2,
            amp1=amp1,
            amp2=amp2,
        ), overwrite=overwrite_file)
    if save_txt:
        with open(sv_root.format(sub) + '.txt', 'w') as outfile:
            json.dump(dict(latency=latency, amplitude=amplitude), outfile)


# %% plot the waveforms and allow interactive peak picking
for sub in subs_run:
    # dims (stim, subs, reps, itds, chs, samples)
    w = w_abr[idx_subs[sub]]
    wr = w_rep[idx_subs[sub]]
    tmin = int((x_min / 1e3 - t[0]) * fs)
    tmax = int((x_max / 1e3 - t[0]) * fs)
    t0 = int(t[0] * fs)

    plt.ion()
    points_clicked = {}
    amplitude = {}
    latency = {}
    y_shift = {}
    x_shift = {}
    y_zoom = {}
    lat1 = {}
    lat2 = {}
    amp1 = {}
    amp2 = {}

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    ax.set_ylabel(r'$\mu$V')
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'sub {sub:02d}')
    counter = 0

    for sh in stim_shifts:
        plt.axvline(sh, color='k', ls=':', lw=1)
        for grid in np.arange(x_min, x_max + 1, 1):
            plt.axvline(sh + grid, color='k', ls=':', lw=1, alpha=0.2)
    plt.xticks(np.arange(x_min, x_min + window * len(stimuli), 2))
    plt.xlim(x_min, x_min + window * len(stimuli))

    for fi, f0 in enumerate(f0s):
        for pi, peak in enumerate(peaks):
            for si, [stim, sh] in enumerate(zip(stimuli, stim_shifts)):
                ri = si // n_typ
                ti = si % n_typ
                reg = regressors[ri]
                typ = types[ti]
                for i in range(n_reps):
                    line, = ax.plot(
                        t_ms[tmin:tmax] + sh,
                        y_display * counter + wr[ri, ti, fi, i, tmin:tmax],
                        color=colors_stim[ti][fi], alpha=0.3, lw=1)
                line, = ax.plot(
                    t_ms[tmin:tmax] + sh,
                    y_display * counter + w[ri, ti, fi, tmin:tmax],
                    color=colors_stim[ti][fi], lw=lw[0], alpha=1,
                    picker=True, pickradius=picker_size[0],
                    label=f'sub{sub:02d}_{reg}_{typ}_{f0}_{peak}')
                text = ax.text(
                    sh, y_display * counter + y_labelshiftprop * y_display,
                    f'{f0}: {peak}', ha='left', va='bottom',
                    color=colors_stim[ti][fi])
                x_shift[line.get_label()] = sh
                y_shift[line.get_label()] = counter * y_display
                y_zoom[line.get_label()] = 1
                amplitude[line.get_label()] = np.nan
                latency[line.get_label()] = np.nan
                lat1[line.get_label()] = np.nan
                lat2[line.get_label()] = np.nan
                amp1[line.get_label()] = np.nan
                amp2[line.get_label()] = np.nan
                # counter += 1
            counter += 1
        counter += 2
    plt.tight_layout()

    fig.canvas.mpl_connect('pick_event', onpick)
    start_time = datetime.datetime.now()
    fig.canvas.mpl_connect('close_event', onclose)
    go_to_next = input('Press enter!')
