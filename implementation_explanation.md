# Implementation Explanation for Report / Presentation

## Pipeline Summary

The implementation follows a simple EEG preprocessing pipeline inspired by the reviewed literature. The goal is not to reproduce the most advanced methods, but to implement a transparent baseline pipeline that can be explained clearly.

## Steps

### 1. Load EEG data

The script supports common EEG file formats through MNE-Python, including EDF, BDF, EEGLAB SET, FIF and BrainVision VHDR.

### 2. Raw time-domain inspection

A short segment of the raw EEG is plotted. This helps visually identify large artifacts such as eye blinks, movement artifacts, flat channels or unusually noisy channels.

### 3. Raw frequency-domain inspection

The script computes and plots the Power Spectral Density (PSD). This helps identify frequency-specific noise, for example power-line noise around 50 Hz.

### 4. Band-pass filtering

A default 1–40 Hz band-pass filter is applied. The high-pass part removes slow baseline drift, while the low-pass part suppresses high-frequency noise such as muscle activity.

### 5. Notch filtering

A notch filter at 50 Hz is applied by default. This targets power-line interference, which is common in Europe.

### 6. Optional ICA

If requested, the script applies ICA. ICA separates EEG into components and can remove components associated with EOG or ECG artifacts if appropriate reference channels are available.

### 7. Before/after evaluation

The script plots the cleaned signal and compares PSD before and after preprocessing. This provides qualitative evidence that artifact-related power has been reduced.

## Why this pipeline is appropriate for LP1

This pipeline is simple, reproducible and explainable. It covers basic artifact-removal concepts from the literature: filtering, notch filtering, source separation and time/frequency-domain validation. Advanced methods such as ASR, deep learning or SOBI-SWT can be discussed as future extensions.
