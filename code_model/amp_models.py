#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:49:13 2021

@author: tom, tong, melissa
"""
# TODO: Correct M1, M3, M5 (probably just scale by 43/401)
# TODO: Test and verify
import numpy as np
import cochlea
from mne.filter import resample
from joblib import Parallel, delayed
import ic_cn2018 as nuclei
import re


def findstring(ref, check):
    r = re.compile("(?:" + "|".join(check) + ")*$")
    if r.match(ref) is not None:
        return True
    return False


def get_rates(stim_up, cf):
    fs_up = int(100e3)  # TODO: this should be passed in to the function
    return np.array(cochlea.run_zilany2014_rate(stim_up,
                                                fs_up,
                                                anf_types='hsr',
                                                cf=cf,
                                                species='human',
                                                cohc=1.,
                                                cihc=1.))[:, 0]


def anm(stim, fs_in, stim_pres_db, parallel=True, n_jobs=-1,
        stim_gen_rms=0.01, cf_low=125, cf_high=16e3, dOct=1/6, shift_ms=1):
    """
    Parameters
    ----------
    stim : ndarray
        A 1-D array containing the stimulus.
    fs_in : flaot
        Sampling rate of the stimulus.
    stim_pres_db : float
        The level at which to present the stimuli to the model.
    parallel : bool, optional
        If True, will use joblib to loop through the model in parallel.
        The default is True.
    n_jobs : int, optional
        Passed to joblib. The maximum number of concurrently running jobs.
        -1 will use all CPU cores. The default is -1.
    stim_gen_rms : float, optional
        The rms used to generate the stimuli. The default is 0.01.
    cf_low : float, optional
        The lowest CF to model. The default is 125.
    cf_high : float, optional
        The highest CF to model. The default is 16e3.
    dOct : float
        spacing of the CF channels
    shift_ms : float
        How much to shift the final output, in miliseconds. Default is 1 ms.
    """

    if fs_in != int(fs_in):
        assert type(fs_in) is int, 'Non-integer fs_in is not yet implemented'
    else:
        fs_in = int(fs_in)

    # Resample your stimuli to a higher fs for the model
    fs_up = int(100e3)
    stim_up = resample(stim, fs_up, fs_in, npad='auto', n_jobs=n_jobs)

    # Put stim in correct units
    # scalar to put it in units of pascals (double-checked and good)
    sine_rms_at_0db = 20e-6
    db_conv = ((sine_rms_at_0db / stim_gen_rms) * 10 ** (stim_pres_db / 20.))
    stim_up = db_conv * stim_up

    # run the model
    cfs = 2 ** np.arange(np.log2(cf_low), np.log2(cf_high + 1), dOct)
    anf_rates_up = np.zeros([len(cfs), len(stim)*fs_up//fs_in])

    if parallel:
        anf_rates_up = Parallel(n_jobs=n_jobs)([delayed(get_rates)(stim_up, cf)
                                                for cf in cfs])
    else:  # might need to debug, never used
        for cfi, cf in enumerate(cfs):
            anf_rates_up[cfi] = get_rates(stim_up, cf)[
                ..., :len(stim) * fs_up//fs_in]  # Off by one for some reason??

    # Downsample to match input fs
    anf_rates = resample(anf_rates_up, fs_in, fs_up, npad='auto',
                         n_jobs=n_jobs)

    # Scale, sum, and shift
    M1 = nuclei.M1
    anm = M1*anf_rates.sum(0)
    final_shift = int(fs_in*shift_ms/1000)
    anm = np.roll(anm, final_shift)
    anm[:final_shift] = anm[final_shift+1]
    return anm


def model_abr(stim, fs_in, fs_out, stim_pres_db, parallel=True, n_jobs=-1,
              stim_gen_rms=0.01, cf_low=125, cf_high=16e3, return_flag='abr'):
    """
    return_flag: str
     Indicates which waves of the abr to return. Defaults to 'abr' which
     returns a single abr waveform containing waves I, III, and V. Can also be
     '1', '3', or '5' to get individual waves. Combining these option will
     return a dict with the desired waveforms. e.g. '13abr' will return a
     dict with keys 'w1', 'w3', and 'abr'
    """

    return_flag = str(return_flag)
    known_flags = ['1', '3', '5', 'abr', 'rates']
    msg = ('return_flag must be a combination of the following: ' +
           str(known_flags))
    assert findstring(return_flag, known_flags), msg

    # Resample your stimuli to a higher fs for the model
    fs_up = int(100e3)
    stim_up = resample(stim, fs_up, fs_in, npad='auto', n_jobs=n_jobs)

    # Put stim in correct units
    # scalar to put it in units of pascals (double-checked and good)
    sine_rms_at_0db = 20e-6
    db_conv = ((sine_rms_at_0db / stim_gen_rms) * 10 ** (stim_pres_db / 20.))
    stim_up = db_conv * stim_up

    # run the model
    dOct = 1. / 6
    cfs = 2 ** np.arange(np.log2(cf_low), np.log2(cf_high + 1), dOct)
    anf_rates_up = np.zeros([len(cfs), len(stim)*fs_up//fs_in])
    n_cf = len(cfs)

    if parallel:
        anf_rates_up = Parallel(n_jobs=n_jobs)([delayed(get_rates)(stim_up, cf)
                                                for cf in cfs])
    else:
        for cfi, cf in enumerate(cfs):
            anf_rates_up[cfi] = get_rates(stim_up, cf)

    # Downsample to match input fs
    anf_rates = resample(anf_rates_up, fs_out, fs_up, npad='auto',
                         n_jobs=n_jobs)

    # sum and filter to get AN and IC response, only use hsf to save time
    w3, w1 = nuclei.cochlearNuclei(anf_rates.T, anf_rates.T, anf_rates.T,
                                   n_cf, 0, 0, fs_out)  # was 1
    # filter to get IC response
    w5 = nuclei.inferiorColliculus(w3, fs_out)

    # shift, scale, and sum responses
    w1_shift = int(fs_out*0.001)
    w3_shift = int(fs_out*0.00225)
    w5_shift = int(fs_out*0.0035)
    w1 = np.roll(np.sum(w1, axis=1)*nuclei.M1, w1_shift)
    w3 = np.roll(np.sum(w3, axis=1)*nuclei.M3, w3_shift)
    w5 = np.roll(np.sum(w5, axis=1)*nuclei.M5, w5_shift)

    # clean up the roll
    w1[:w1_shift] = w1[w1_shift+1]
    w3[:w3_shift] = w3[w3_shift+1]
    w5[:w5_shift] = w5[w5_shift+1]

    scale = 401 / n_cf

    # Handle output
    if return_flag == 'abr':
        return w1+w3+w5

    waves = {}
    if 'abr' in return_flag:
        waves['abr'] = w1+w3+w5
    if '1' in return_flag:
        waves['w1'] = w1
    if '3' in return_flag:
        waves['w3'] = w3
    if '5' in return_flag:
        waves['w5'] = w5
    if 'rates' in return_flag:
        waves['rates'] = anf_rates

    return waves, scale
