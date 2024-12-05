# peaky_pitchshift code
This repository is the code for the article "Fundamental frequency predominantly drives talker differences in auditory brainstem responses to continuous speech" by Polonenko & Maddox (2024).

Jasa-EL: Melissa J. Polonenko, Ross K. Maddox; Fundamental frequency predominantly drives talker differences in auditory brainstem responses to continuous speech. JASA Express Lett. 1 November 2024; 4 (11): 114401. https://doi.org/10.1121/10.0034329

BioRxiv: 2024.07.12.603125; doi: https://doi.org/10.1101/2024.07.12.603125

The EEG-BIDS format data for the paper are available on OpenNeuro: [https://openneuro.org/datasets/ds005340/versions/1.0.4]

### Abbreviations
- f0: fundamental frequency
- ABR: auditory brainstem response

### Pre-requisite libraries
- `mne`
- `expyfun`
- `parselmouth`
- `cochlea`

## Code descriptions
Code files are listed in order that they were used in the pipeline.

### Stimulus pre-processing
code_stimuli folder
- `code_pitchshift_USE.py`: determining f0s and factors to pitch shift stimuli using PRAAT through `parselmouth`
- `make_clicks.py`: code parameters to make randomized click trains, calls on `pip_trains_rme.py`
- `pip_trains_rme.py`: creates the clicks
- `wave_t_pulses_bands_USE.py`: creates peaky speech, calls on `wav_to_pitch.praat` and `wav_to_pulses.praat` using PRAAT through `parselmouth`
- `wav_to_pulses.praat`: code to determine pulse trains in the speech wave file
- `wav_to_pitch.praat`: code to determine the f0 of the speech wave file
- `combine_slices`: combines 1s trials into 10s trials
- `stimulus_isi.py`: code to determine inter-stimulus interval (isi) distributions for the 6 speech conditions and 3 click conditions

### Experiment code
code_experiment folder
- `exp_peaky_pitch_pilot.py`: ran a pilot experiment on one person
- `exp_peaky_pitch.py`: the main code to run the EEG experiment using `expyfun`

### Modeling code
code_model folder
- `create_regressors.py`: creates anm model regressor (not used for the paper, although available with the dataset)
- `create_model.py`: simulates EEG data from the wav files, finds optimal shift and factor values to match lowest-rate click modeled ABR to measured ABR and uses those to create simulated EEG for all stimuli, calls on `amp_models.py` and `ic_cn2018.py`
- `amp_models.py`: code to run Zilany 2014 model, calls on `ic_cn2018.py` and requires the `cochlea` package (https://pypi.org/project/cochlea/)
- `ic_cn2018.py`: available from Verhult github (https://github.com/HearingTechnology/Verhulstetal2018Model)
PBS scripts are included, which was used to run the model and regressor scripts on CLA compute

### Analysis code
code_analysis folder
- `power_analyses.Rmd`: determine the number of participants required for the study; based on Polonenko & Maddox (2021) https://elifesciences.org/articles/62329
- `rename_stimuli.py`: renamed for ease of compliance to EEG-BIDS format
- `eeg_bids.py`: convert raw EEG data to EEG-BIDS format and include relevant files
- `analysis_eeg.py`: pre-process EEG and derive responses for each participant
- `peak_picker_code_pitch2.py`: interactive code to pick peaks in the ABRs, gives an hdf5 file with all the chosen peak amplitudes and latencies (earlier version `peak_picker_code_pitch.py`)
- `peak_picker_code_pitch2_model.py`: modified peak-picker code for the modeled ABRs
- `compile_peaks.py`: read in the files output from the peak picker code and organize in a useable format for the formal analyses
- `analysis_plotting.py`: the main file for statistics and plotting data for the paper
- `statistics_lmer.Rmd`: the R markdown file used to run the linear mixed effects model for the wave V peak amplitudes and latencies

  
