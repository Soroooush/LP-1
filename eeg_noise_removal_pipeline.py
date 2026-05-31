from __future__ import annotations

import argparse
from pathlib import Path
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt

try:
    import mne
except ImportError as e:
    raise SystemExit(
        "MNE-Python is required. Install it with: pip install mne\n"
        f"Original error: {e}"
    )


def load_eeg(input_path: Path):
    """Load EEG data based on file extension."""
    suffix = input_path.suffix.lower()

    if suffix == ".fif":
        raw = mne.io.read_raw_fif(input_path, preload=True, verbose=False)
    elif suffix == ".edf":
        raw = mne.io.read_raw_edf(input_path, preload=True, verbose=False)
    elif suffix == ".bdf":
        raw = mne.io.read_raw_bdf(input_path, preload=True, verbose=False)
    elif suffix == ".set":
        raw = mne.io.read_raw_eeglab(input_path, preload=True, verbose=False)
    elif suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(input_path, preload=True, verbose=False)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. Try .edf, .bdf, .set, .fif, or .vhdr."
        )

    # Try to make channel types EEG if they are unknown.
    unknown = [ch for ch, typ in zip(raw.ch_names, raw.get_channel_types()) if typ == "misc"]
    if unknown:
        raw.set_channel_types({ch: "eeg" for ch in unknown})

    return raw


def save_basic_report(raw, output_dir: Path, filename: str = "metadata_report.json"):
    """Save basic metadata as JSON."""
    info = {
        "n_channels": len(raw.ch_names),
        "channel_names": raw.ch_names,
        "channel_types": raw.get_channel_types(),
        "sampling_frequency_hz": float(raw.info["sfreq"]),
        "duration_seconds": float(raw.times[-1]) if len(raw.times) else None,
        "reference": str(raw.info.get("custom_ref_applied", "unknown")),
    }
    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def plot_raw_segment(raw, output_dir: Path, title: str, filename: str, duration: float = 10.0):
    """Plot a short time-domain segment from selected EEG channels."""
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, exclude="bads")
    if len(eeg_picks) == 0:
        warnings.warn("No EEG channels found for raw segment plot.")
        return

    picks = eeg_picks[: min(8, len(eeg_picks))]
    sfreq = raw.info["sfreq"]
    n_samples = int(min(duration * sfreq, raw.n_times))
    data, times = raw[picks, :n_samples]

    # Convert from volts to microvolts if MNE data are in volts.
    data_uv = data * 1e6

    plt.figure(figsize=(12, 6))
    offset = 0
    spacing = np.nanstd(data_uv) * 6 if np.nanstd(data_uv) > 0 else 50
    for i, ch_data in enumerate(data_uv):
        plt.plot(times, ch_data + offset, linewidth=0.8, label=raw.ch_names[picks[i]])
        offset += spacing
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude + offset (µV)")
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200)
    plt.close()


def plot_psd(raw, output_dir: Path, title: str, filename: str, fmin: float = 0.5, fmax: float = 80.0):
    """Plot PSD for EEG channels."""
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, exclude="bads")
    if len(eeg_picks) == 0:
        warnings.warn("No EEG channels found for PSD plot.")
        return

    spectrum = raw.compute_psd(picks=eeg_picks, fmin=fmin, fmax=fmax, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    mean_psd = psds.mean(axis=0)

    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, mean_psd)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power spectral density")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200)
    plt.close()


def apply_basic_filters(raw, l_freq: float, h_freq: float, notch_freq: float | None):
    """Apply band-pass and optional notch filter."""
    cleaned = raw.copy()

    # Band-pass filtering: removes slow drift and high-frequency noise.
    cleaned.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose=False)

    # Notch filtering: useful for power-line noise, e.g. 50 Hz in Europe.
    if notch_freq is not None and notch_freq > 0:
        cleaned.notch_filter(freqs=[notch_freq], picks="eeg", verbose=False)

    return cleaned


def run_optional_ica(raw, output_dir: Path, n_components: float | int = 0.95, random_state: int = 42):
    """
    Run optional ICA for artifact-component inspection/removal.

    This is conservative: it tries automatic EOG/ECG detection only if channels exist.
    If no EOG/ECG channels exist, it saves ICA component plots for manual inspection.
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, exclude="bads")
    if len(eeg_picks) < 2:
        warnings.warn("ICA skipped: at least 2 EEG channels are required.")
        return raw, {"ica_applied": False, "reason": "fewer than 2 EEG channels"}

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="fastica",
        random_state=random_state,
        max_iter="auto",
        verbose=False,
    )
    ica.fit(raw, picks=eeg_picks, verbose=False)

    exclude = []
    detection = {"eog_components": [], "ecg_components": []}

    # Automatic EOG detection if EOG channel exists.
    if "eog" in raw.get_channel_types():
        try:
            eog_inds, eog_scores = ica.find_bads_eog(raw, verbose=False)
            exclude.extend(eog_inds)
            detection["eog_components"] = [int(x) for x in eog_inds]
        except Exception as e:
            detection["eog_error"] = str(e)

    # Automatic ECG detection if ECG channel exists.
    if "ecg" in raw.get_channel_types():
        try:
            ecg_inds, ecg_scores = ica.find_bads_ecg(raw, verbose=False)
            exclude.extend(ecg_inds)
            detection["ecg_components"] = [int(x) for x in ecg_inds]
        except Exception as e:
            detection["ecg_error"] = str(e)

    exclude = sorted(set(exclude))
    ica.exclude = exclude

    # Save component overview for manual inspection.
    try:
        figs = ica.plot_components(show=False)
        if not isinstance(figs, list):
            figs = [figs]
        for i, fig in enumerate(figs):
            fig.savefig(output_dir / f"ica_components_{i+1}.png", dpi=200)
            plt.close(fig)
    except Exception as e:
        detection["component_plot_error"] = str(e)

    cleaned = raw.copy()
    if exclude:
        ica.apply(cleaned, verbose=False)
        detection["ica_applied"] = True
        detection["excluded_components"] = [int(x) for x in exclude]
    else:
        detection["ica_applied"] = False
        detection["reason"] = "No automatic EOG/ECG components detected; inspect component plots manually."

    return cleaned, detection


def main():
    parser = argparse.ArgumentParser(description="Simple EEG noise-removal pipeline in Python/MNE.")
    parser.add_argument("--input", required=True, help="Path to EEG file: .edf, .bdf, .set, .fif, .vhdr")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--l_freq", type=float, default=1.0, help="High-pass cutoff in Hz, default 1.0")
    parser.add_argument("--h_freq", type=float, default=40.0, help="Low-pass cutoff in Hz, default 40.0")
    parser.add_argument("--notch", type=float, default=50.0, help="Notch frequency in Hz, default 50. Use 0 to disable.")
    parser.add_argument("--run_ica", action="store_true", help="Run optional ICA artifact-component removal")
    parser.add_argument("--save_cleaned", action="store_true", help="Save cleaned data as .fif")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    notch = None if args.notch == 0 else args.notch

    print(f"Loading EEG: {input_path}")
    raw = load_eeg(input_path)
    raw_original = raw.copy()

    print("Saving metadata and pre-cleaning figures...")
    metadata = save_basic_report(raw, output_dir)
    plot_raw_segment(raw, output_dir, "Raw EEG: time-domain segment", "01_raw_time_domain.png")
    plot_psd(raw, output_dir, "Raw EEG: average PSD", "02_raw_psd.png")

    print(f"Applying filters: band-pass {args.l_freq}-{args.h_freq} Hz, notch={notch}")
    cleaned = apply_basic_filters(raw, l_freq=args.l_freq, h_freq=args.h_freq, notch_freq=notch)

    ica_report = {"ica_requested": bool(args.run_ica)}
    if args.run_ica:
        print("Running optional ICA...")
        cleaned, ica_report = run_optional_ica(cleaned, output_dir)

    print("Saving post-cleaning figures...")
    plot_raw_segment(cleaned, output_dir, "Cleaned EEG: time-domain segment", "03_cleaned_time_domain.png")
    plot_psd(cleaned, output_dir, "Cleaned EEG: average PSD", "04_cleaned_psd.png")

    # Save comparison plot for PSD before/after.
    eeg_picks = mne.pick_types(raw_original.info, eeg=True, eog=False, ecg=False, exclude="bads")
    if len(eeg_picks) > 0:
        psd_raw = raw_original.compute_psd(picks=eeg_picks, fmin=0.5, fmax=min(80, raw_original.info['sfreq']/2 - 1), verbose=False)
        psd_clean = cleaned.compute_psd(picks=eeg_picks, fmin=0.5, fmax=min(80, cleaned.info['sfreq']/2 - 1), verbose=False)
        raw_vals, freqs = psd_raw.get_data(return_freqs=True)
        clean_vals, freqs2 = psd_clean.get_data(return_freqs=True)
        plt.figure(figsize=(10, 5))
        plt.semilogy(freqs, raw_vals.mean(axis=0), label="Raw")
        plt.semilogy(freqs2, clean_vals.mean(axis=0), label="Cleaned")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power spectral density")
        plt.title("PSD comparison: raw vs cleaned EEG")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "05_psd_raw_vs_cleaned.png", dpi=200)
        plt.close()

    if args.save_cleaned:
        cleaned_path = output_dir / "cleaned_eeg_raw.fif"
        cleaned.save(cleaned_path, overwrite=True, verbose=False)
        print(f"Saved cleaned EEG: {cleaned_path}")

    report = {
        "input_file": str(input_path),
        "metadata": metadata,
        "pipeline": {
            "bandpass_hz": [args.l_freq, args.h_freq],
            "notch_hz": notch,
            "ica": ica_report,
        },
        "outputs": [
            "01_raw_time_domain.png",
            "02_raw_psd.png",
            "03_cleaned_time_domain.png",
            "04_cleaned_psd.png",
            "05_psd_raw_vs_cleaned.png",
        ],
    }
    with open(output_dir / "pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Done. Results saved to:", output_dir)


if __name__ == "__main__":
    main()
