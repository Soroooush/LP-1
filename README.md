LP1 EEG Noise-Removal Pipeline
This repository contains the implementation part of my LP Praktikum Informatik 1 project on EEG noise and artifact removal.
The project demonstrates a simple but explainable EEG preprocessing workflow in Python using MNE-Python. The implementation was developed after reviewing literature on EEG artifact removal and was extended to include dataset justification, filtering, re-referencing, synthetic artifact examples, bad-channel detection, bad-channel interpolation, and ICA-based blink removal.
---
Project Overview
The goal of this project is to demonstrate important steps in EEG preprocessing:
loading real EEG data,
inspecting raw EEG in the time domain and frequency domain,
applying basic filtering and re-referencing,
creating controlled synthetic artifact examples,
detecting bad channels using signal-quality metrics,
detecting bad channel-epoch pairs,
interpolating persistent bad channels,
demonstrating ICA-based blink-artifact removal,
saving figures and pipeline reports.
The implementation is intended as a transparent baseline pipeline rather than a full clinical or production-level EEG-cleaning system.
---
Dataset
The EEG file used in the notebook is:
```text
S001R01.edf
```
This file comes from the PhysioNet EEG Motor Movement/Imagery Dataset, also known as EEGMMIDB.
Dataset details:
File: `S001R01.edf`
Subject: Subject 1
Run: Run 1
Experiment: baseline eyes-open recording
Channels: 64 EEG channels
Sampling frequency: 160 Hz
Duration: approximately 61 seconds
This file was chosen because it is public, small enough for demonstration, compatible with MNE-Python, and provides realistic multichannel EEG data.
Dataset Limitation
The original `S001R01.edf` recording does not contain labeled eye-blink, muscle, or bad-channel artifacts. Therefore, controlled synthetic artifacts were added in the notebook to demonstrate artifact detection and removal steps.
Synthetic examples include:
blink-like artifact in frontal channels,
muscle-like high-frequency noise in temporal channels,
flat channel example,
noisy bad-channel example.
---
Main Notebook
Use this notebook:
```text
eeg_noise_removal_pipeline_REPLACEMENT_epochwise.ipynb
```
This is the final replacement notebook containing the complete implementation.
---
Repository Structure
Recommended structure:
```text
.
├── data/
│   └── S001R01.edf
├── eeg_noise_removal_pipeline_REPLACEMENT_epochwise.ipynb
├── results/
├── README.md
└── requirements.txt
```
The `data/` folder should contain the EEG file:
```text
data/S001R01.edf
```
---
Pipeline Summary
The notebook follows this workflow:
```text
Load EEG data
→ Raw EEG inspection
→ PSD inspection
→ 1–40 Hz band-pass filtering
→ 50 Hz notch filtering
→ Common Average Reference
→ Synthetic artifact examples
→ Global bad-channel detection
→ Epoch-wise bad-channel / bad-epoch detection
→ Persistent bad-channel selection
→ Bad-channel interpolation
→ ICA blink-component removal
→ Save output figures and reports
```
---
Implemented Steps
1. Raw EEG Inspection
The notebook first visualizes the original EEG data:
time-domain EEG segment,
average Power Spectral Density, PSD.
This helps identify general signal quality, line noise, high-frequency noise, and possible abnormal channels.
---
2. Basic Preprocessing
The basic preprocessing includes:
1 Hz high-pass filter to reduce slow drift,
40 Hz low-pass filter to reduce high-frequency noise,
50 Hz notch filter as a standard power-line-noise step,
Common Average Reference, CAR to re-reference all EEG channels.
The filter range was chosen as a simple baseline because many standard EEG rhythms are mainly within 1–40 Hz.
---
3. Synthetic Artifact Demonstration
Because the original dataset does not provide labeled artifact examples, the notebook adds controlled synthetic artifacts:
blink-like slow wave in frontal channels,
muscle-like high-frequency noise in temporal channels,
flat channel,
noisy channel.
This makes it possible to demonstrate artifact detection, interpolation, and ICA removal in a controlled way.
---
4. Global Bad-Channel Detection
The notebook computes channel-quality metrics across the whole recording:
standard deviation,
peak-to-peak amplitude,
mean correlation with other channels,
robust z-scores based on median and MAD.
The robust z-score formula is:
```text
robust z-score = 0.6745 × (value − median) / MAD
```
Detected suspicious channels include channels with unusually high noise, unusually large amplitude range, or very low/flat activity.
---
5. Epoch-Wise Bad-Channel Detection
The EEG is split into fixed-length 2-second epochs.
For each channel in each epoch, the notebook computes:
standard deviation,
peak-to-peak amplitude,
correlation with the median signal across channels.
A channel is marked bad in an epoch if it is unusually flat, noisy, high-amplitude, or poorly correlated with the rest of the EEG.
An entire epoch is marked bad if more than 20% of channels are bad in that epoch.
---
6. Persistent Bad Channels
A persistent bad channel is defined as a channel that is bad in more than 30% of epochs.
This prevents interpolating channels that are only temporarily affected by artifacts. Persistent bad channels are better candidates for interpolation.
---
7. Bad-Channel Interpolation
Detected persistent or structural bad channels are interpolated using MNE-Python.
Interpolation estimates the signal of a bad channel from neighboring good electrodes. This is useful for examples such as:
flat channels,
extremely noisy channels.
---
8. ICA Blink-Removal Example
The notebook demonstrates ICA-based blink-artifact removal:
a synthetic blink is added to frontal channels,
ICA is fitted to the EEG data,
the component most correlated with the frontal blink signal is selected,
the blink-like component is removed,
the EEG is reconstructed and compared before/after.
ICA is demonstrated mainly for eye-blink artifacts because blink artifacts are spatially stereotyped and easier to isolate than many muscle artifacts.
---
Output Files
Running the notebook creates output files in:
```text
results/
```
Typical outputs include:
```text
01_original_raw_time.png
02_original_raw_psd.png
03_basic_preprocessed_time.png
04_original_vs_basic_psd.png
05_synthetic_artifacts_time.png
06_synthetic_artifacts_psd.png
07_before_interpolation.png
08_after_interpolation.png
09_blink_like_ica_component.png
10_before_ica_blink.png
11_after_ica_blink.png
bad_channel_metrics_global.json
bad_channel_epoch_metrics.json
bad_channel_epoch_heatmap.png
persistent_bad_channels.json
replacement_pipeline_report.json
```
---
Requirements
Install the required Python packages:
```bash
pip install mne numpy matplotlib
```
Recommended optional packages:
```bash
pip install jupyter pandas scipy
```
---
How to Run
Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```
Create a `data/` folder:
```bash
mkdir data
```
Place the EEG file inside the folder:
```text
data/S001R01.edf
```
Open the notebook:
```text
eeg_noise_removal_pipeline_REPLACEMENT_epochwise.ipynb
```
Run all cells from top to bottom.
The notebook will create:
```text
results/
```
containing figures and JSON reports.
---
Notes on Data Sharing
If the EEG file is large or should not be redistributed, do not commit the `data/` folder to GitHub.
Recommended `.gitignore` entries:
```gitignore
data/
results/
*.edf
*.fif
*.png
*.json
.ipynb_checkpoints/
```
If the dataset is not included in the repository, users should download or place `S001R01.edf` manually in the `data/` folder.
---
Limitations
The used file is a baseline eyes-open recording and does not contain labeled EOG, EMG, or bad-channel artifact annotations.
Synthetic artifacts are simplified examples used for controlled demonstration.
ICA is demonstrated for blink-like artifacts, not as a complete automatic artifact-removal solution.
The pipeline is educational and explainable, not optimized for clinical or production EEG preprocessing.
---
Summary
This project demonstrates an end-to-end EEG preprocessing workflow:
```text
inspect → filter / re-reference → add controlled artifacts → detect → interpolate / remove → compare
```
The final notebook connects the literature review to a practical Python/MNE implementation and shows how basic EEG preprocessing steps can be applied, visualized, and documented.
