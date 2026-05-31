# LP1 EEG Noise-Removal Pipeline

This folder contains a simple Python/MNE implementation for the LP Praktikum Informatik 1 EEG artifact-removal task.

## Goal

Implement a simple, explainable EEG preprocessing pipeline:

1. Load EEG data
2. Plot raw EEG in the time domain
3. Plot raw EEG PSD in the frequency domain
4. Apply band-pass filtering
5. Apply notch filtering for 50 Hz power-line noise
6. Optionally apply ICA for EOG/ECG-like components
7. Compare raw vs cleaned EEG using PSD and time-domain plots
8. Save a processing report

## Installation

```bash
pip install mne numpy matplotlib
```

## Example commands

```bash
python eeg_noise_removal_pipeline.py --input data/sample.edf --output results --l_freq 1 --h_freq 40 --notch 50
```

With optional ICA:

```bash
python eeg_noise_removal_pipeline.py --input data/sample.edf --output results --l_freq 1 --h_freq 40 --notch 50 --run_ica --save_cleaned
```

## Output files

The script creates:

- `01_raw_time_domain.png`
- `02_raw_psd.png`
- `03_cleaned_time_domain.png`
- `04_cleaned_psd.png`
- `05_psd_raw_vs_cleaned.png`
- `metadata_report.json`
- `pipeline_report.json`
- optionally: `cleaned_eeg_raw.fif`

## Interpretation

A successful basic cleaning should show:

- reduced slow drift after high-pass filtering
- reduced high-frequency noise after low-pass filtering
- reduced 50 Hz peak after notch filtering
- more stable PSD after preprocessing

## Notes

This is intentionally a simple baseline pipeline. More advanced methods from the literature include ASR, deep learning denoising, SOBI + machine learning + SWT, and full pipelines such as NEAR.

