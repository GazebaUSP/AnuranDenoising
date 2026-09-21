"""
Audio denoising pipeline for anuran (frog) call recordings.

Given raw field recordings, this pipeline:
  1. standardizes the audio (sample rate, mono, normalization) [standardize_audio],
  2. removes human speech segments (VAD) [remove_human_speech],
  3. band-pass filters the signal [bandpass],
  4. detects the dominant frequency band(s) of the calls [detect_dominant_band],
  5. detects "call" frames vs. noise/silence [detect_call_frames],
  6. expands note boundaries (Schmitt-trigger-like hysteresis) [expand_notes],
  7. applies stationary noise reduction [reduce_noise_noisereduce],
  8. applies several bin-level and note-level cleanup/filtering stages
     [median_reduction, filter_bins_by_correlation_and_probability,
     count_bins_per_frame_and_entropy, note_duration, centroid_analysis,
     mean_continuous_band_length, deexpand_erroneously_long_notes,
     eliminate_weak_pseudo_notes],
  9. trims silence from the cleaned audio [trim_clean_audio],
  10. saves intermediate audio files and diagnostic spectrogram plots for every
      stage [process_file / _process_file_internal].

All heavy processing works on the magnitude of the Short-Time Fourier Transform
(STFT); the original phase is kept and re-applied (via inverse STFT) whenever a
cleaned audio file needs to be written to disk.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.lines
import os
import shutil
import tempfile
import stat
from concurrent.futures import ProcessPoolExecutor
import traceback
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt
from silero_vad import get_speech_timestamps, load_silero_vad
import noisereduce
import soundfile as sf
from scipy.ndimage import uniform_filter1d


def standardize_audio(audio_path: str) -> tuple:
    """
    Load an audio file, resample it to a fixed target sample rate, convert it
    to mono, and peak-normalize its amplitude to [-1, 1].

    Args:
        audio_path: Path to the input audio file (any format librosa can read).

    Returns:
        A tuple (standardized_audio, sample_rate):
            standardized_audio: float32 mono waveform, peak-normalized.
            sample_rate: the target sample rate used (Hz).
    """
    target_sample_rate = 44100
    standardized_audio, sample_rate = librosa.load(audio_path, sr=target_sample_rate, mono=True)
    standardized_audio = standardized_audio.astype(np.float32)
    standardized_audio /= np.max(np.abs(standardized_audio))
    return standardized_audio, sample_rate


def remove_human_speech(standardized_audio: np.ndarray, sample_rate: int, file_name: str) -> tuple:
    """
    Detect human speech segments with Silero VAD and remove them from the audio.

    The VAD model requires 16 kHz audio, so the input is resampled just for
    detection; the returned (speech-free) audio keeps the original sample rate.

    Args:
        standardized_audio: Mono waveform (output of standardize_audio).
        sample_rate: Sample rate of standardized_audio (Hz).
        file_name: File name, used only for logging.

    Returns:
        A tuple (audio_after_vad, speech_timestamps, speech_mask):
            audio_after_vad: waveform with detected speech segments removed.
            speech_timestamps: list of dicts with "start"/"end" (seconds) for
                each detected speech segment, as returned by Silero VAD.
            speech_mask: boolean mask (same length as standardized_audio) that
                is True at every sample classified as human speech.
    """
    vad_model = load_silero_vad()
    audio_16khz = librosa.resample(standardized_audio, orig_sr=sample_rate, target_sr=16000)
    speech_timestamps = get_speech_timestamps(audio_16khz, vad_model, sampling_rate=16000, threshold=0.8, min_speech_duration_ms=4000, min_silence_duration_ms=4000, speech_pad_ms=500, return_seconds=True)
    del audio_16khz
    print(f"  [{file_name}] VAD done | {len(speech_timestamps)} speech segments removed")
    speech_mask = np.zeros(len(standardized_audio), dtype=bool)
    for speech_segment in speech_timestamps:
        speech_start_sample = int(float(speech_segment["start"]) * sample_rate)
        speech_end_sample    = int(float(speech_segment["end"])   * sample_rate)
        speech_mask[speech_start_sample:speech_end_sample] = True
    audio_after_vad = standardized_audio[~speech_mask].astype(np.float32)
    return audio_after_vad, speech_timestamps, speech_mask


lowpass_frequency  = 12000
highpass_frequency = 300
def bandpass(audio_after_vad: np.ndarray, sample_rate: int, file_name: str) -> np.ndarray:
    """
    Apply a 4th-order Butterworth high-pass filter followed by a 4th-order
    Butterworth low-pass filter (zero-phase, via sosfiltfilt), restricting the
    signal to the [highpass_frequency, lowpass_frequency] band.

    Args:
        audio_after_vad: Waveform to filter (typically the speech-removed audio).
        sample_rate: Sample rate of audio_after_vad (Hz).
        file_name: File name, used only for logging.

    Returns:
        The band-pass filtered waveform (float32).
    """
    sos_highpass = butter(4, highpass_frequency / (0.5 * float(sample_rate)), btype="highpass", output="sos")
    sos_lowpass  = butter(4, lowpass_frequency  / (0.5 * float(sample_rate)), btype="lowpass",  output="sos")
    filtered_audio = sosfiltfilt(sos_lowpass, sosfiltfilt(sos_highpass, audio_after_vad)).astype(np.float32)
    print(f"  [{file_name}] high-pass and low-pass filters done")
    return filtered_audio


def detect_dominant_band(spectrum_magnitude: np.ndarray, time_seconds: np.ndarray, fft_freqs: np.ndarray) -> tuple:
    """
    Detect the dominant frequency band(s) of the call across the whole recording.

    For each STFT frame, the bin with the highest magnitude is taken as that
    frame's "dominant bin". Bins are then scored by how much total (weighted)
    energy they accumulate as a dominant bin across all frames. Starting from
    the highest-energy bins (seed bands where the normalized accumulated
    energy exceeds initial_band_energy_threshold), candidate frames are
    iteratively attached to the nearest eligible band: a candidate is
    eligible only if it lies within max_multiplier_distance_threshold_hz of
    the band, or within one band-width, AND its magnitude exceeds a
    distance-dependent statistical threshold (mean minus a dynamic multiple
    of the standard deviation, computed over that band's frames) that grows
    stricter the farther the candidate is from the band. When several bands
    are eligible, the candidate is attached to the closest one. Finally,
    bands that end up close to each other (closer than band_merge_distance_hz)
    are merged.

    Args:
        spectrum_magnitude: STFT magnitude, shape (n_freq_bins, n_frames).
        time_seconds: Time (s) of each STFT frame, shape (n_frames,).
        fft_freqs: Frequency (Hz) of each STFT bin, shape (n_freq_bins,).

    Returns:
        A tuple (dominant_band_min_frequencies, dominant_band_max_frequencies, plots):
            dominant_band_min_frequencies: array with the lower edge (Hz) of
                each detected dominant band.
            dominant_band_max_frequencies: array with the upper edge (Hz) of
                each detected dominant band.
            plots: list of 2 diagnostic plotting functions (each takes a
                matplotlib Axes and draws on it).
    """
    initial_band_energy_threshold = 0.2
    band_distance_tolerance_factor = 1
    band_merge_distance_hz = 200
    magnitude_exponent = 2
    min_dominant_band_magnitude_threshold = 0.01
    max_std_multiplier = 0.1
    min_std_multiplier = -10
    min_multiplier_ratio = 0.5
    max_multiplier_ratio = band_distance_tolerance_factor
    max_multiplier_distance_threshold_hz = 75
    multiplier_decay_exponent = 2

    absolute_energy_per_frame = (spectrum_magnitude ** 2).sum(axis=0)
    frames_with_energy_mask = absolute_energy_per_frame > 0
    spectrum_magnitude = spectrum_magnitude[:, frames_with_energy_mask]
    time_seconds_with_energy = time_seconds[frames_with_energy_mask]

    dominant_bin_per_frame = np.argmax(spectrum_magnitude, axis=0)
    dominant_frequency_per_frame = fft_freqs[dominant_bin_per_frame]
    dominant_magnitude_per_frame = spectrum_magnitude[dominant_bin_per_frame, np.arange(spectrum_magnitude.shape[1])]
    normalized_dominant_magnitude = dominant_magnitude_per_frame / (dominant_magnitude_per_frame.max() + 1e-12)

    full_audio_bins_mask = (fft_freqs >= dominant_frequency_per_frame.min()) & (fft_freqs <= dominant_frequency_per_frame.max())
    full_audio_bins = np.flatnonzero(full_audio_bins_mask)
    energy_per_dominant_bin = np.bincount(dominant_bin_per_frame, weights=dominant_magnitude_per_frame ** (magnitude_exponent + 1), minlength=fft_freqs.size)
    accumulated_energy_full_audio_bins = energy_per_dominant_bin[full_audio_bins]
    normalized_accumulated_energy = accumulated_energy_full_audio_bins / (accumulated_energy_full_audio_bins.max() + 1e-12)

    initial_band_bin_mask = normalized_accumulated_energy > initial_band_energy_threshold
    initial_band_bin_indices = np.flatnonzero(initial_band_bin_mask)
    breaks_between_initial_bands = np.where(np.diff(initial_band_bin_indices) > 1)[0]
    current_band_starts = np.concatenate(([initial_band_bin_indices[0]], initial_band_bin_indices[breaks_between_initial_bands + 1])).tolist() if initial_band_bin_indices.size > 0 else []
    current_band_ends = np.concatenate((initial_band_bin_indices[breaks_between_initial_bands], [initial_band_bin_indices[-1]])).tolist() if initial_band_bin_indices.size > 0 else []

    initial_bands_label = ", ".join(f"{fft_freqs[full_audio_bins[start]]:.0f}-{fft_freqs[full_audio_bins[end]]:.0f} Hz" for start, end in zip(current_band_starts, current_band_ends))

    local_bin_index_per_frame = np.searchsorted(full_audio_bins, dominant_bin_per_frame)
    frames_already_included_mask = np.zeros(local_bin_index_per_frame.size, dtype=bool)
    for band_start, band_end in zip(current_band_starts, current_band_ends):
        frames_already_included_mask |= (local_bin_index_per_frame >= band_start) & (local_bin_index_per_frame <= band_end)

    while True:
        candidate_attached_this_iteration = False
        for candidate_frame_index in np.flatnonzero(~frames_already_included_mask):
            candidate_frequency = dominant_frequency_per_frame[candidate_frame_index]
            candidate_magnitude = normalized_dominant_magnitude[candidate_frame_index]
            best_band_index, smallest_candidate_band_distance = None, np.inf
            for band_index, (band_start, band_end) in enumerate(zip(current_band_starts, current_band_ends)):
                band_start_frequency, band_end_frequency = fft_freqs[full_audio_bins[band_start]], fft_freqs[full_audio_bins[band_end]]
                current_band_width = band_end_frequency - band_start_frequency
                candidate_band_distance = max(0.0, band_start_frequency - candidate_frequency, candidate_frequency - band_end_frequency)
                frames_within_band_mask = (local_bin_index_per_frame >= band_start) & (local_bin_index_per_frame <= band_end) & (normalized_dominant_magnitude > min_dominant_band_magnitude_threshold)
                magnitudes_within_band = normalized_dominant_magnitude[frames_within_band_mask]
                distance_width_ratio = candidate_band_distance / current_band_width if current_band_width > 0 else np.inf
                clamped_multiplier_ratio = min(max(distance_width_ratio, min_multiplier_ratio), max_multiplier_ratio)
                dynamic_std_multiplier = max_std_multiplier + ((clamped_multiplier_ratio - min_multiplier_ratio) / (max_multiplier_ratio - min_multiplier_ratio)) ** multiplier_decay_exponent * (min_std_multiplier - max_std_multiplier)
                dynamic_std_multiplier = max(dynamic_std_multiplier, max_std_multiplier) if candidate_band_distance <= max_multiplier_distance_threshold_hz else dynamic_std_multiplier
                band_magnitude_threshold = magnitudes_within_band.mean() - dynamic_std_multiplier * magnitudes_within_band.std()
                if (candidate_band_distance <= max_multiplier_distance_threshold_hz or candidate_band_distance < band_distance_tolerance_factor * current_band_width) and candidate_magnitude > max(min_dominant_band_magnitude_threshold, band_magnitude_threshold) and candidate_band_distance < smallest_candidate_band_distance:
                    best_band_index, smallest_candidate_band_distance = band_index, candidate_band_distance
            if best_band_index is not None:
                candidate_local_bin_index = local_bin_index_per_frame[candidate_frame_index]
                current_band_starts[best_band_index] = min(current_band_starts[best_band_index], candidate_local_bin_index)
                current_band_ends[best_band_index] = max(current_band_ends[best_band_index], candidate_local_bin_index)
                frames_already_included_mask |= (local_bin_index_per_frame >= current_band_starts[best_band_index]) & (local_bin_index_per_frame <= current_band_ends[best_band_index])
                candidate_attached_this_iteration = True
        if not candidate_attached_this_iteration:
            break

    sorted_current_band_indices = sorted(range(len(current_band_starts)), key=lambda index: current_band_starts[index])
    merged_band_starts = []
    merged_band_ends = []
    for band_index in sorted_current_band_indices:
        band_start, band_end = current_band_starts[band_index], current_band_ends[band_index]
        if merged_band_starts and fft_freqs[full_audio_bins[band_start]] - fft_freqs[full_audio_bins[merged_band_ends[-1]]] < band_merge_distance_hz:
            merged_band_ends[-1] = max(merged_band_ends[-1], band_end)
        else:
            merged_band_starts.append(band_start)
            merged_band_ends.append(band_end)
    kept_band_starts = np.array(merged_band_starts, dtype=int)
    kept_band_ends = np.array(merged_band_ends, dtype=int)

    final_bands_label = ", ".join(f"{fft_freqs[full_audio_bins[start_index]]:.0f}-{fft_freqs[full_audio_bins[end_index]]:.0f} Hz" for start_index, end_index in zip(kept_band_starts, kept_band_ends))

    dominant_band_min_frequencies = fft_freqs[full_audio_bins[kept_band_starts]]
    dominant_band_max_frequencies = fft_freqs[full_audio_bins[kept_band_ends]]

    def plot_kept_bands_accumulated_energy(ax) -> None:
        bar_width = fft_freqs[full_audio_bins[1]] - fft_freqs[full_audio_bins[0]]
        ax.bar(fft_freqs[full_audio_bins], normalized_accumulated_energy, width=bar_width, color="black", label=f"Initial bands: {initial_bands_label}\nFinal bands: {final_bands_label}")
        ax.axhline(initial_band_energy_threshold, color="blue", linewidth=1.0, linestyle="--")
        ax.set_xlim(highpass_frequency, fft_freqs[full_audio_bins[-1]])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Accumulated energy weighted by max magnitude, peak-normalized")
        ax.set_title("Accumulated energy weighted by max magnitude per bin, whole audio")
        ax.legend(loc="upper right", fontsize=8)

    def plot_dominant_band_by_accumulated_energy(ax) -> None:
        subsampling_step = max(1, len(time_seconds_with_energy) // 6000)
        ax.scatter(time_seconds_with_energy[::subsampling_step], dominant_frequency_per_frame[::subsampling_step], c=normalized_dominant_magnitude[::subsampling_step], cmap="gray_r", s=0.5, linewidths=0.2, alpha=0.7, vmin=0, vmax=1)
        for band_min_frequency, band_max_frequency in zip(dominant_band_min_frequencies, dominant_band_max_frequencies):
            label = f"{band_min_frequency:.0f}-{band_max_frequency:.0f} Hz"
            ax.axhline(band_min_frequency, color="red", linewidth=0.3, label=label)
            ax.axhline(band_max_frequency, color="red", linewidth=0.3)
        ax.set_xlim(time_seconds[0], time_seconds[-1])
        ax.set_ylim(0, lowpass_frequency)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Dominant frequency (Hz)")
        ax.set_title("Dominant bands by accumulated energy")
        ax.legend(loc="upper right", fontsize=8)

    return dominant_band_min_frequencies, dominant_band_max_frequencies, [plot_kept_bands_accumulated_energy, plot_dominant_band_by_accumulated_energy]


def detect_call_frames(spectrum_magnitude: np.ndarray, sample_rate: int, time_seconds: np.ndarray, dominant_band_min_frequencies: np.ndarray, dominant_band_max_frequencies: np.ndarray, fft_freqs: np.ndarray) -> tuple:
    """
    Classify each STFT frame as containing a call (croak) or not, based on
    how much of its energy sits inside the dominant band(s) and how loud it is.

    A frame is approved if either:
      - its normalized max magnitude is high AND its spectral concentration
        in the dominant band is at least moderate; or
      - its spectral concentration in the dominant band is very high AND its
        normalized max magnitude is at least low.
    This lets loud-but-slightly-off-band frames pass, and quiet-but-very-
    concentrated frames pass too.

    Args:
        spectrum_magnitude: STFT magnitude, shape (n_freq_bins, n_frames).
        sample_rate: Sample rate of the audio (Hz).
        time_seconds: Time (s) of each STFT frame, shape (n_frames,).
        dominant_band_min_frequencies: Lower edge(s) (Hz) of the dominant
            band(s), as returned by detect_dominant_band.
        dominant_band_max_frequencies: Upper edge(s) (Hz) of the dominant
            band(s), as returned by detect_dominant_band.
        fft_freqs: Frequency (Hz) of each STFT bin, shape (n_freq_bins,).

    Returns:
        A tuple (spectral_concentration, spectrum_magnitude_after_call_detection,
        approved_frame_mask, max_magnitude_per_frame, plots):
            spectral_concentration: fraction of each frame's power that falls
                inside the dominant band(s), shape (n_frames,).
            spectrum_magnitude_after_call_detection: spectrum_magnitude with
                every rejected frame zeroed out.
            approved_frame_mask: boolean mask, True for frames classified as
                containing a call.
            max_magnitude_per_frame: raw (non-normalized) max magnitude of
                each frame.
            plots: list of 3 diagnostic plotting functions.
    """
    high_magnitude_threshold = 0.3
    mid_spectral_concentration_threshold = 0.4
    low_magnitude_threshold = 0.09
    high_spectral_concentration_threshold = 0.75

    dominant_band_mask = np.zeros_like(fft_freqs, dtype=bool)
    for band_min_frequency, band_max_frequency in zip(dominant_band_min_frequencies, dominant_band_max_frequencies):
        dominant_band_mask |= (fft_freqs >= band_min_frequency) & (fft_freqs <= band_max_frequency)

    dominant_band_power_per_frame = (spectrum_magnitude[dominant_band_mask, :] ** 2).sum(axis=0)
    total_power_per_frame = (spectrum_magnitude ** 2).sum(axis=0)
    spectral_concentration = dominant_band_power_per_frame / (total_power_per_frame + 1e-12)
    peak_spectral_concentration = spectral_concentration.max()
    absolute_mid_spectral_concentration_threshold = mid_spectral_concentration_threshold * peak_spectral_concentration
    absolute_high_spectral_concentration_threshold = high_spectral_concentration_threshold * peak_spectral_concentration

    max_magnitude_per_frame = spectrum_magnitude.max(axis=0)
    normalized_max_magnitude = max_magnitude_per_frame / (max_magnitude_per_frame.max() + 1e-12)

    approved_frame_mask = ((normalized_max_magnitude > high_magnitude_threshold) & (spectral_concentration > absolute_mid_spectral_concentration_threshold)) | ((spectral_concentration > absolute_high_spectral_concentration_threshold) & (normalized_max_magnitude > low_magnitude_threshold))
    spectrum_magnitude_after_call_detection = spectrum_magnitude * approved_frame_mask[None, :]

    def plot_max_magnitude_over_time(ax) -> None:
        ax.plot(time_seconds, normalized_max_magnitude, color="black", linewidth=0.8)
        ax.axhline(high_magnitude_threshold, color="red", linewidth=1.0, linestyle="--")
        ax.axhline(low_magnitude_threshold, color="blue", linewidth=1.0, linestyle="--")
        ax.set_xlim(time_seconds[0], time_seconds[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frame normalized max magnitude")
        ax.set_title("Max magnitude per frame over time")

    def plot_spectral_concentration(ax) -> None:
        ax.plot(time_seconds, spectral_concentration, color="black", linewidth=0.8)
        ax.axhline(absolute_mid_spectral_concentration_threshold, color="red", linewidth=1.0, linestyle="--")
        ax.axhline(absolute_high_spectral_concentration_threshold, color="blue", linewidth=1.0, linestyle="--")
        ax.set_xlim(time_seconds[0], time_seconds[-1])
        ax.set_ylim(0, peak_spectral_concentration)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spectral concentration in the dominant band")
        ax.set_title("Spectral concentration per frame over time")

    def plot_spectrogram_after_call_detection(ax) -> None:
        subsampling_step = max(1, spectrum_magnitude_after_call_detection.shape[1] // 1000)
        max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude.shape[0] - 1)) + 1
        spectrum_magnitude_after_call_detection_db = librosa.amplitude_to_db(spectrum_magnitude_after_call_detection[:max_frequency_bin_index, ::subsampling_step], ref=spectrum_magnitude.max())
        ax.imshow(spectrum_magnitude_after_call_detection_db, origin="lower", aspect="auto", extent=[time_seconds[0], time_seconds[-1], 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title(f"Spectrogram after VAD | approved frames: {approved_frame_mask.mean() * 100:.1f}%")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    return spectral_concentration, spectrum_magnitude_after_call_detection, approved_frame_mask, max_magnitude_per_frame, [plot_max_magnitude_over_time, plot_spectral_concentration, plot_spectrogram_after_call_detection]


def expand_notes(spectrum_magnitude: np.ndarray, spectrum: np.ndarray, approved_frame_mask: np.ndarray, sample_rate: int, time_seconds: np.ndarray, dominant_band_min_frequencies: np.ndarray, dominant_band_max_frequencies: np.ndarray, hop_length: int, file_name: str, notes_output_path: str, audio_length_samples: int, fft_freqs: np.ndarray) -> tuple:
    """
    Expand the boundaries of the already-approved call frames (notes) using a
    Schmitt-trigger-like hysteresis, driven by bins whose power over time is
    correlated with a narrow reference frequency band.

    Steps:
      1. Build a "reference band" around the weighted-mean frequency of the
         dominant band(s) (mean +/- weighted std / std_divisor_factor).
      2. For every bin inside the band-pass range, cross-correlate (over a
         range of frame lags) its per-frame energy with the reference band's
         per-frame power, and keep the max correlation across lags. Bins
         whose max correlation is >= correlation_threshold are considered
         well correlated with the call's temporal envelope.
      3. Smooth those bins' magnitude over time (moving average).
      4. From each already-approved note's start/end frame, walk outward
         frame by frame; a frame is added to the note (Schmitt "on") as long
         as the smoothed magnitude of at least one still-active bin keeps
         falling relative to the last frame that grew the note; the walk
         stops once no active bin is still falling.
      5. Prune isolated single-frame gaps: if the whole gap between two
         consecutive notes ended up included after expansion, the single
         weakest frame in that gap is set back to False, keeping the notes
         separated.
      6. Reconstruct the audio (inverse STFT with the original phase) for
         the expanded notes and write it to disk.

    Args:
        spectrum_magnitude: STFT magnitude, shape (n_freq_bins, n_frames).
        spectrum: Complex STFT (magnitude and phase), same shape.
        approved_frame_mask: Boolean mask of frames approved by
            detect_call_frames.
        sample_rate: Sample rate of the audio (Hz).
        time_seconds: Time (s) of each STFT frame.
        dominant_band_min_frequencies: Lower edge(s) (Hz) of the dominant band(s).
        dominant_band_max_frequencies: Upper edge(s) (Hz) of the dominant band(s).
        hop_length: STFT hop length (samples), needed for istft.
        file_name: File name, used only for logging.
        notes_output_path: Path to write the reconstructed audio to.
        audio_length_samples: Target length (samples) for the reconstructed audio.
        fft_freqs: Frequency (Hz) of each STFT bin.

    Returns:
        A tuple (audio_after_schmitt, schmitt_spectrum_magnitude, post_schmitt_mask,
        notes, max_correlation_per_bin, plots):
            audio_after_schmitt: reconstructed waveform after note expansion.
            schmitt_spectrum_magnitude: spectrum_magnitude masked by
                post_schmitt_mask.
            post_schmitt_mask: boolean frame mask after expansion and gap pruning.
            notes: list of (start_frame, end_frame) tuples for each final note.
            max_correlation_per_bin: max cross-correlation (over lags) for each
                band-pass bin, used by later stages.
            plots: list of 2 diagnostic plotting functions.
    """
    std_divisor_factor = 3
    correlation_threshold = 0.75
    max_lag_frames = 50
    moving_average_window_size = 10

    bandpass_mask = (fft_freqs >= highpass_frequency) & (fft_freqs <= lowpass_frequency)
    bandpass_fft_freqs = fft_freqs[bandpass_mask]
    bandpass_bin_indices = np.flatnonzero(bandpass_mask)

    dominant_band_mask = np.zeros_like(fft_freqs, dtype=bool)
    for band_min_frequency, band_max_frequency in zip(dominant_band_min_frequencies, dominant_band_max_frequencies):
        dominant_band_mask |= (fft_freqs >= band_min_frequency) & (fft_freqs <= band_max_frequency)

    accumulated_magnitude_dominant_band_bins = spectrum_magnitude[dominant_band_mask, :].sum(axis=1).astype(np.float64)
    dominant_band_fft_freqs = fft_freqs[dominant_band_mask]
    weighted_mean_dominant_band_frequency = (dominant_band_fft_freqs * accumulated_magnitude_dominant_band_bins).sum() / accumulated_magnitude_dominant_band_bins.sum()
    weighted_std_dominant_band_frequency = max(np.sqrt((accumulated_magnitude_dominant_band_bins * (dominant_band_fft_freqs - weighted_mean_dominant_band_frequency) ** 2).sum() / accumulated_magnitude_dominant_band_bins.sum()) / std_divisor_factor, fft_freqs[1] - fft_freqs[0])
    reference_band_mask = (fft_freqs >= weighted_mean_dominant_band_frequency - weighted_std_dominant_band_frequency) & (fft_freqs <= weighted_mean_dominant_band_frequency + weighted_std_dominant_band_frequency)

    energy_per_bin_per_frame = (spectrum_magnitude[bandpass_mask, :] ** 2).astype(np.float64, copy=False)
    reference_band_power_per_frame = (spectrum_magnitude[reference_band_mask, :] ** 2).sum(axis=0).astype(np.float64, copy=False)

    total_frame_count = reference_band_power_per_frame.size
    frame_lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    bandpass_bin_count = energy_per_bin_per_frame.shape[0]
    correlation_per_bin_per_lag = np.zeros((bandpass_bin_count, frame_lags.size), dtype=np.float64)
    for lag_index, lag in enumerate(frame_lags):
        reference_band_segment = reference_band_power_per_frame[lag:] if lag >= 0 else reference_band_power_per_frame[:total_frame_count + lag]
        per_bin_segment = energy_per_bin_per_frame[:, :total_frame_count - lag] if lag >= 0 else energy_per_bin_per_frame[:, -lag:]
        centered_reference_band = reference_band_segment - reference_band_segment.mean()
        centered_bin = per_bin_segment - per_bin_segment.mean(axis=1, keepdims=True)
        numerator = centered_bin @ centered_reference_band
        denominator = np.sqrt((centered_bin ** 2).sum(axis=1) * (centered_reference_band ** 2).sum())
        correlation_per_bin_per_lag[:, lag_index] = numerator / (denominator + 1e-12)
    max_correlation_per_bin = correlation_per_bin_per_lag.max(axis=1)

    approved_bin_indices = bandpass_bin_indices[max_correlation_per_bin >= correlation_threshold]
    approved_bins_spectrum_magnitude = spectrum_magnitude[approved_bin_indices, :]
    smoothed_approved_bins_spectrum_magnitude = uniform_filter1d(approved_bins_spectrum_magnitude, size=moving_average_window_size, axis=1, mode="nearest")

    approved_frame_indices = np.flatnonzero(approved_frame_mask)
    breaks_between_notes = np.where(np.diff(approved_frame_indices) > 1)[0]
    note_starts = np.concatenate(([approved_frame_indices[0]], approved_frame_indices[breaks_between_notes + 1])) if approved_frame_indices.size > 0 else np.array([], dtype=int)
    note_ends = np.concatenate((approved_frame_indices[breaks_between_notes], [approved_frame_indices[-1]])) if approved_frame_indices.size > 0 else np.array([], dtype=int)

    post_schmitt_mask = approved_frame_mask.copy()
    for note_start, note_end in zip(note_starts, note_ends):
        for edge_index, direction in ((note_start, -1), (note_end, 1)):
            active_bins_mask = np.ones(smoothed_approved_bins_spectrum_magnitude.shape[0], dtype=bool)
            tracked_magnitude_per_bin = smoothed_approved_bins_spectrum_magnitude[:, edge_index].copy()
            neighbor_frame_index = edge_index + direction
            while 0 <= neighbor_frame_index < smoothed_approved_bins_spectrum_magnitude.shape[1] and active_bins_mask.any():
                neighbor_magnitude_per_bin = smoothed_approved_bins_spectrum_magnitude[:, neighbor_frame_index]
                bins_still_falling_mask = active_bins_mask & (neighbor_magnitude_per_bin < tracked_magnitude_per_bin)
                if not bins_still_falling_mask.any():
                    break
                post_schmitt_mask[neighbor_frame_index] = True
                tracked_magnitude_per_bin[bins_still_falling_mask] = neighbor_magnitude_per_bin[bins_still_falling_mask]
                active_bins_mask &= bins_still_falling_mask
                neighbor_frame_index += direction

    approved_bins_frame_magnitude = smoothed_approved_bins_spectrum_magnitude.sum(axis=0)
    for next_note_start, previous_note_end in zip(note_starts[1:], note_ends[:-1]):
        gap_frame_indices = np.arange(previous_note_end + 1, next_note_start)
        if gap_frame_indices.size > 0 and post_schmitt_mask[gap_frame_indices].all():
            weakest_frame_index = gap_frame_indices[np.argmin(approved_bins_frame_magnitude[gap_frame_indices])]
            post_schmitt_mask[weakest_frame_index] = False

    post_schmitt_frame_indices = np.flatnonzero(post_schmitt_mask)
    post_schmitt_breaks_between_notes = np.where(np.diff(post_schmitt_frame_indices) > 1)[0]
    final_note_starts = np.concatenate(([post_schmitt_frame_indices[0]], post_schmitt_frame_indices[post_schmitt_breaks_between_notes + 1])) if post_schmitt_frame_indices.size > 0 else np.array([], dtype=int)
    final_note_ends = np.concatenate((post_schmitt_frame_indices[post_schmitt_breaks_between_notes], [post_schmitt_frame_indices[-1]])) if post_schmitt_frame_indices.size > 0 else np.array([], dtype=int)
    notes = list(zip(final_note_starts.tolist(), final_note_ends.tolist()))

    schmitt_spectrum_magnitude = spectrum_magnitude * post_schmitt_mask[None, :]
    original_phase = np.angle(spectrum)
    audio_after_schmitt = librosa.istft(schmitt_spectrum_magnitude * np.exp(1j * original_phase), hop_length=hop_length, window="hann", center=True, length=audio_length_samples).astype(np.float32)
    sf.write(notes_output_path, audio_after_schmitt, sample_rate)
    print(f"  [{file_name}] audio after Schmitt expansion saved to {notes_output_path}")

    def plot_reference_cross_correlation(ax) -> None:
        ax.plot(bandpass_fft_freqs, max_correlation_per_bin, color="black", linewidth=0.8)
        ax.axhline(correlation_threshold, color="red", linewidth=1.0, linestyle="--")
        ax.set_xlim(bandpass_fft_freqs[0], bandpass_fft_freqs[-1])
        ax.set_ylim(min(0, max_correlation_per_bin.min()), 1)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Max correlation (best lag)")
        ax.set_title(f"Cross-correlation between each bin's power and the weighted mean+/-std/{std_divisor_factor} band power ({weighted_mean_dominant_band_frequency:.0f} Hz +/- {weighted_std_dominant_band_frequency:.0f} Hz)")

    def plot_spectrogram_after_expansion(ax) -> None:
        subsampling_step = max(1, schmitt_spectrum_magnitude.shape[1] // 1000)
        max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude.shape[0] - 1)) + 1
        schmitt_spectrum_magnitude_db = librosa.amplitude_to_db(schmitt_spectrum_magnitude[:max_frequency_bin_index, ::subsampling_step], ref=spectrum_magnitude.max())
        approved_frame_percentage = post_schmitt_mask.mean() * 100
        re_approved_frame_percentage = (post_schmitt_mask & ~approved_frame_mask).mean() * 100
        ax.imshow(schmitt_spectrum_magnitude_db, origin="lower", aspect="auto", extent=[time_seconds[0], time_seconds[-1], 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title(f"STFT after expansion | approved: {approved_frame_percentage:.1f}% | re-approved: {re_approved_frame_percentage:.1f}%")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    return audio_after_schmitt, schmitt_spectrum_magnitude, post_schmitt_mask, notes, max_correlation_per_bin, [plot_reference_cross_correlation, plot_spectrogram_after_expansion]


def reduce_noise_noisereduce(spectrum_magnitude: np.ndarray, spectrum: np.ndarray, post_schmitt_mask: np.ndarray, audio_after_schmitt: np.ndarray, sample_rate: int, n_fft: int, hop_length: int, file_name: str, noise_output_path: str, audio_length_samples: int) -> tuple:
    """
    Apply stationary noise reduction (via the `noisereduce` package) to the
    audio, using the frames rejected by the Schmitt expansion (i.e. the
    frames without a call) as the noise profile.

    Args:
        spectrum_magnitude: STFT magnitude of the original audio, used only
            as a reference for dB conversion (ref=spectrum_magnitude.max()).
        spectrum: Complex STFT of the audio after Schmitt expansion, used to
            isolate the noise-only frames.
        post_schmitt_mask: Boolean frame mask from expand_notes (True = call).
        audio_after_schmitt: Waveform after Schmitt expansion.
        sample_rate: Sample rate of the audio (Hz).
        n_fft: FFT size used for STFT/ISTFT.
        hop_length: STFT hop length (samples).
        file_name: File name, used only for logging.
        noise_output_path: Path to write the noise-reduced audio to.
        audio_length_samples: Target length (samples) for the output audio.

    Returns:
        A tuple (noise_reduced_audio, spectrum_after_reduction,
        spectrum_magnitude_after_reduction, plots):
            noise_reduced_audio: waveform after noisereduce.
            spectrum_after_reduction: complex STFT of noise_reduced_audio.
            spectrum_magnitude_after_reduction: magnitude of spectrum_after_reduction.
            plots: list of 2 diagnostic plotting functions.
    """
    stationary                    = True
    prop_decrease                 = 1.0
    n_std_thresh_stationary       = 1
    freq_mask_smooth_hz           = 50
    time_mask_smooth_ms           = 25

    noise_frames_exist = bool((~post_schmitt_mask).any())
    isolated_noise_audio = librosa.istft(spectrum[:, ~post_schmitt_mask], hop_length=hop_length, window="hann", center=True).astype(np.float32) if noise_frames_exist else None

    noise_reduced_audio = librosa.util.fix_length(noisereduce.reduce_noise(y=audio_after_schmitt, sr=sample_rate, y_noise=isolated_noise_audio, stationary=stationary, prop_decrease=prop_decrease, n_std_thresh_stationary=n_std_thresh_stationary, freq_mask_smooth_hz=freq_mask_smooth_hz, time_mask_smooth_ms=time_mask_smooth_ms, n_fft=n_fft, hop_length=hop_length).astype(np.float32), size=audio_length_samples)
    sf.write(noise_output_path, noise_reduced_audio, sample_rate)
    print(f"  [{file_name}] audio after noisereduce saved to {noise_output_path}" if noise_frames_exist else f"  [{file_name}] no noise frames isolated (all frames survived Schmitt expansion) | audio after noisereduce saved to {noise_output_path}")

    spectrum_after_reduction = librosa.stft(noise_reduced_audio, n_fft=n_fft, hop_length=hop_length, window="hann", center=True)
    spectrum_magnitude_after_reduction = np.abs(spectrum_after_reduction).astype(np.float32, copy=False)

    if noise_frames_exist:
        isolated_noise_magnitude = np.abs(librosa.stft(isolated_noise_audio, n_fft=n_fft, hop_length=hop_length, window="hann", center=True))
        isolated_noise_magnitude_db = librosa.amplitude_to_db(isolated_noise_magnitude, ref=spectrum_magnitude.max())
        noise_mean_per_frequency = isolated_noise_magnitude_db.mean(axis=1)
        noise_std_per_frequency = isolated_noise_magnitude_db.std(axis=1)
        noise_threshold_per_frequency = noise_mean_per_frequency + n_std_thresh_stationary * noise_std_per_frequency
        threshold_spectrogram = np.full_like(spectrum_magnitude, -80.0)
        threshold_spectrogram[:, post_schmitt_mask] = noise_threshold_per_frequency[:, np.newaxis]
    else:
        threshold_spectrogram = np.full_like(spectrum_magnitude, -80.0)

    def plot_noise_threshold_surviving_frames(ax) -> None:
        subsampling_step = max(1, threshold_spectrogram.shape[1] // 1000)
        max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude.shape[0] - 1)) + 1
        ax.imshow(threshold_spectrogram[:max_frequency_bin_index, ::subsampling_step], origin="lower", aspect="auto", extent=[0, threshold_spectrogram.shape[1] * hop_length / sample_rate, 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title(f"Noise threshold µ+{n_std_thresh_stationary}σ projected onto the frames surviving Schmitt expansion" if noise_frames_exist else "No noise frames isolated (Schmitt expansion classified every frame as a call)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    def plot_spectrogram_after_noisereduce(ax) -> None:
        subsampling_step = max(1, spectrum_magnitude_after_reduction.shape[1] // 1000)
        max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude_after_reduction.shape[0] - 1)) + 1
        spectrum_magnitude_after_noisereduce_db = librosa.amplitude_to_db(spectrum_magnitude_after_reduction[:max_frequency_bin_index, ::subsampling_step], ref=spectrum_magnitude_after_reduction.max())
        ax.imshow(spectrum_magnitude_after_noisereduce_db, origin="lower", aspect="auto", extent=[0, spectrum_magnitude_after_reduction.shape[1] * hop_length / sample_rate, 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title(f"Spectrogram after noisereduce (stationary={stationary}, prop_decrease={prop_decrease})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    return noise_reduced_audio, spectrum_after_reduction, spectrum_magnitude_after_reduction, [plot_noise_threshold_surviving_frames, plot_spectrogram_after_noisereduce]


def median_reduction(spectrum_magnitude: np.ndarray, spectrum_magnitude_after_reduction: np.ndarray, phase_after_reduction: np.ndarray, post_schmitt_mask: np.ndarray, sample_rate: int, hop_length: int, file_name: str, median_output_path: str, audio_length_samples: int) -> tuple:
    """
    Apply a per-bin noise gate based on median absolute deviation (MAD):
    for each frequency bin, estimate a noise power threshold from the median
    and MAD of that bin's power across the noise-only frames (those rejected
    by the Schmitt expansion), then zero out that bin, in call frames, at
    any point where its power (post-noisereduce) does not exceed
    median + mad_multiplier * MAD.

    Args:
        spectrum_magnitude: STFT magnitude of the original audio (used to
            estimate the per-bin noise threshold from its noise frames).
        spectrum_magnitude_after_reduction: STFT magnitude after
            reduce_noise_noisereduce.
        phase_after_reduction: Phase (radians) of the STFT after
            reduce_noise_noisereduce, used for reconstruction.
        post_schmitt_mask: Boolean frame mask from expand_notes (True = call).
        sample_rate: Sample rate of the audio (Hz).
        hop_length: STFT hop length (samples).
        file_name: File name, used only for logging.
        median_output_path: Path to write the reconstructed audio to.
        audio_length_samples: Target length (samples) for the output audio.

    Returns:
        A tuple (audio_after_median, spectrum_magnitude_after_median, plots):
            audio_after_median: reconstructed waveform after the median gate.
            spectrum_magnitude_after_median: spectrum_magnitude_after_reduction
                with below-threshold bins zeroed in call frames.
            plots: list with 1 diagnostic plotting function.
    """
    mad_multiplier = 3

    original_spectrum_power = spectrum_magnitude ** 2
    spectrum_power_after_reduction = spectrum_magnitude_after_reduction ** 2
    noise_frames_exist = bool((~post_schmitt_mask).any())

    if noise_frames_exist:
        noise_power_per_bin = original_spectrum_power[:, ~post_schmitt_mask]
        noise_median_per_bin = np.median(noise_power_per_bin, axis=1)
        noise_mad_per_bin = np.median(np.abs(noise_power_per_bin - noise_median_per_bin[:, None]), axis=1)
        power_threshold_per_bin = noise_median_per_bin + mad_multiplier * noise_mad_per_bin
    else:
        power_threshold_per_bin = np.zeros(spectrum_power_after_reduction.shape[0], dtype=np.float64)

    call_spectrum_power = spectrum_power_after_reduction * post_schmitt_mask[None, :]
    bin_above_threshold_mask_per_frame = call_spectrum_power > power_threshold_per_bin[:, None]
    frame_count_per_bin_above_threshold = bin_above_threshold_mask_per_frame.sum(axis=1)
    spectrum_magnitude_after_median = spectrum_magnitude_after_reduction * bin_above_threshold_mask_per_frame

    audio_after_median = librosa.istft(spectrum_magnitude_after_median * np.exp(1j * phase_after_reduction), hop_length=hop_length, window="hann", center=True, length=audio_length_samples).astype(np.float32)
    sf.write(median_output_path, audio_after_median, sample_rate)
    print(f"  [{file_name}] audio after median reduction saved to {median_output_path}" if noise_frames_exist else f"  [{file_name}] no noise frames isolated (all frames survived Schmitt expansion) | audio after median reduction saved to {median_output_path}")

    def plot_spectrogram_after_median(ax) -> None:
        subsampling_step = max(1, spectrum_magnitude_after_median.shape[1] // 1000)
        max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude_after_median.shape[0] - 1)) + 1
        spectrum_magnitude_after_median_db = librosa.amplitude_to_db(spectrum_magnitude_after_median[:max_frequency_bin_index, ::subsampling_step], ref=spectrum_magnitude_after_reduction.max())
        ax.imshow(spectrum_magnitude_after_median_db, origin="lower", aspect="auto", extent=[0, spectrum_magnitude_after_median.shape[1] * hop_length / sample_rate, 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title("Spectrogram after median reduction")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    return audio_after_median, spectrum_magnitude_after_median, [plot_spectrogram_after_median]


def filter_bins_by_correlation_and_probability(spectrum_magnitude: np.ndarray, spectrum_magnitude_after_median: np.ndarray, phase_after_reduction: np.ndarray, sample_rate: int, hop_length: int, file_name: str, correlation_noise_filter_output_path: str, audio_length_samples: int, fft_freqs: np.ndarray, max_correlation_per_bin_note_expansion: np.ndarray) -> tuple:
    """
    Approve or reject each frequency bin using an OR of three criteria, then
    zero out rejected bins in every frame.

    A bin is approved if any of the following holds:
      1. It is a "clean" bin: the percentage of noise-only frames whose power
         is at or below that bin's median power (measured across signal
         frames) exceeds noise_below_median_percentage_threshold — i.e. the
         bin is usually much quieter in noise than in signal.
      2. Its max cross-correlation (carried over from expand_notes) exceeds
         an absolute high threshold.
      3. Its correlation is only intermediate, but it lies within
         max_nearby_high_correlation_bin_distance_hz of some other bin that
         does meet the high-correlation criterion.
    Only band-pass bins that already have some energy in
    spectrum_magnitude_after_median (above bin_present_magnitude_threshold)
    are evaluated; bins that never show up there are left untouched (they
    are already zero).

    Args:
        spectrum_magnitude: STFT magnitude of the original audio, used to
            estimate each bin's power distribution across noise-only frames.
        spectrum_magnitude_after_median: STFT magnitude after median_reduction.
        phase_after_reduction: Phase (radians) used for reconstruction.
        sample_rate: Sample rate of the audio (Hz).
        hop_length: STFT hop length (samples).
        file_name: File name, used only for logging.
        correlation_noise_filter_output_path: Path to write the reconstructed
            audio to.
        audio_length_samples: Target length (samples) for the output audio.
        fft_freqs: Frequency (Hz) of each STFT bin.
        max_correlation_per_bin_note_expansion: Max correlation per band-pass
            bin, as returned by expand_notes.

    Returns:
        A tuple (spectrum_magnitude_after_correlation_noise_filter, plots):
            spectrum_magnitude_after_correlation_noise_filter: spectrum with
                rejected bins zeroed out.
            plots: list of 2 diagnostic plotting functions.
    """
    bin_present_magnitude_threshold = 0.0001
    noise_below_median_percentage_threshold = 25
    approval_correlation_threshold = 0.55
    intermediate_correlation_threshold = 0.3
    max_nearby_high_correlation_bin_distance_hz = 75

    signal_frame_mask = spectrum_magnitude_after_median.sum(axis=0) > 0
    original_spectrum_power = spectrum_magnitude ** 2
    spectrum_power_after_median = spectrum_magnitude_after_median ** 2

    bin_present_mask = (fft_freqs >= highpass_frequency) & (fft_freqs <= lowpass_frequency) & (spectrum_magnitude_after_median > bin_present_magnitude_threshold).any(axis=1)
    present_fft_freqs = fft_freqs[bin_present_mask]
    present_filtered_spectrum_power = spectrum_power_after_median[bin_present_mask, :]
    present_original_spectrum_power = original_spectrum_power[bin_present_mask, :]
    signal_frames_bin_power = present_filtered_spectrum_power[:, signal_frame_mask]
    noise_frames_bin_power = present_original_spectrum_power[:, ~signal_frame_mask]
    median_power_per_bin = np.median(signal_frames_bin_power, axis=1)
    noise_below_median_percentage_per_bin = (noise_frames_bin_power <= median_power_per_bin[:, None]).mean(axis=1) * 100 if noise_frames_bin_power.shape[1] > 0 else np.full(present_fft_freqs.size, 100.0)

    full_bandpass_mask = (fft_freqs >= highpass_frequency) & (fft_freqs <= lowpass_frequency)
    full_max_correlation_per_bin = np.zeros_like(fft_freqs)
    full_max_correlation_per_bin[full_bandpass_mask] = max_correlation_per_bin_note_expansion
    present_max_correlation_per_bin = full_max_correlation_per_bin[bin_present_mask]

    peak_max_correlation_per_bin = present_max_correlation_per_bin.max()
    absolute_approval_correlation_threshold = approval_correlation_threshold * peak_max_correlation_per_bin
    absolute_intermediate_correlation_threshold = intermediate_correlation_threshold * peak_max_correlation_per_bin
    high_correlation_bin_mask = present_max_correlation_per_bin > absolute_approval_correlation_threshold
    bin_distance_matrix_hz = np.abs(present_fft_freqs[:, None] - present_fft_freqs[None, :])
    near_high_correlation_bin_mask = (bin_distance_matrix_hz <= max_nearby_high_correlation_bin_distance_hz) @ high_correlation_bin_mask.astype(int) > 0
    intermediate_correlation_bin_mask = (present_max_correlation_per_bin > absolute_intermediate_correlation_threshold) & ~high_correlation_bin_mask & near_high_correlation_bin_mask

    approved_bin_mask = (noise_below_median_percentage_per_bin > noise_below_median_percentage_threshold) | high_correlation_bin_mask | intermediate_correlation_bin_mask

    spectrum_magnitude_after_correlation_noise_filter = spectrum_magnitude_after_median.copy()
    spectrum_magnitude_after_correlation_noise_filter[bin_present_mask, :] *= approved_bin_mask[:, None]

    audio_after_correlation_noise_filter = librosa.istft(spectrum_magnitude_after_correlation_noise_filter * np.exp(1j * phase_after_reduction), hop_length=hop_length, window="hann", center=True, length=audio_length_samples).astype(np.float32)
    sf.write(correlation_noise_filter_output_path, audio_after_correlation_noise_filter, sample_rate)
    print(f"  [{file_name}] audio after noise-percentage/correlation OR filter saved to {correlation_noise_filter_output_path} | {(~approved_bin_mask).sum()}/{approved_bin_mask.size} bins rejected")

    def plot_noise_below_median_percentage_per_bin(ax) -> None:
        bar_colors = np.where(approved_bin_mask, "black", "red")
        ax.bar(present_fft_freqs, noise_below_median_percentage_per_bin, width=present_fft_freqs[1] - present_fft_freqs[0], color=bar_colors)
        ax.axhline(noise_below_median_percentage_threshold, color="blue", linewidth=1.0, linestyle="--")
        ax.set_xlim(present_fft_freqs[0], present_fft_freqs[-1])
        ax.set_ylim(0, 105)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("P: % of noise frames with power <= the bin's median power in signal frames")
        ax.set_title(f"Approval: P > {noise_below_median_percentage_threshold}% OR correlation > {absolute_approval_correlation_threshold:.3f} OR (correlation > {absolute_intermediate_correlation_threshold:.3f} and within {max_nearby_high_correlation_bin_distance_hz} Hz of a bin > {absolute_approval_correlation_threshold:.3f}) ({approved_bin_mask.sum()}/{approved_bin_mask.size} bins approved)")
        ax.legend(handles=[matplotlib.patches.Patch(color="black", label="Approved bin"), matplotlib.patches.Patch(color="red", label="Rejected bin")], loc="upper right", fontsize=8)

    def plot_spectrogram_after_correlation_noise_filter(ax) -> None:
        subsampling_step = max(1, spectrum_magnitude_after_correlation_noise_filter.shape[1] // 1000)
        max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude_after_correlation_noise_filter.shape[0] - 1)) + 1
        spectrum_magnitude_after_correlation_noise_filter_db = librosa.amplitude_to_db(spectrum_magnitude_after_correlation_noise_filter[:max_frequency_bin_index, ::subsampling_step], ref=spectrum_magnitude_after_median.max())
        ax.imshow(spectrum_magnitude_after_correlation_noise_filter_db, origin="lower", aspect="auto", extent=[0, spectrum_magnitude_after_correlation_noise_filter.shape[1] * hop_length / sample_rate, 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title(f"Spectrogram after distribution filter | {approved_bin_mask.sum()}/{approved_bin_mask.size} bins approved")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    return spectrum_magnitude_after_correlation_noise_filter, [plot_noise_below_median_percentage_per_bin, plot_spectrogram_after_correlation_noise_filter]


def count_bins_per_frame_and_entropy(spectrum_magnitude_after_correlation_noise_filter: np.ndarray, phase_after_reduction: np.ndarray, notes: list, sample_rate: int, hop_length: int, file_name: str, count_output_path: str, audio_length_samples: int, fft_freqs: np.ndarray) -> tuple:
    """
    Approve or reject each frame (within the previously detected notes) based
    on how many band-pass bins are "active" in it and, as a secondary check,
    how spectrally concentrated (low-entropy) the frame is.

    For every band-pass bin, a per-bin magnitude threshold is set at
    present_bin_median_threshold_factor times that bin's median magnitude
    over the whole recording; a frame's "bin count" is how many bins exceed
    their own threshold in that frame. Frames within notes are then approved
    if either:
      - their bin count is >= auto_approval_count_fraction of the peak
        in-note bin count (clearly rich in active bins); or
      - their bin count is in an intermediate range (between
        conditional_approval_count_fraction and auto_approval_count_fraction
        of the peak) AND their spectral entropy is low enough (i.e. energy
        concentrated in few bins), judged via a MAD-based threshold on the
        entropic concentration (1 - entropy / max possible entropy).

    Args:
        spectrum_magnitude_after_correlation_noise_filter: STFT magnitude
            after filter_bins_by_correlation_and_probability.
        phase_after_reduction: Phase (radians) used for reconstruction.
        notes: List of (start_frame, end_frame) tuples, as returned by
            expand_notes.
        sample_rate: Sample rate of the audio (Hz).
        hop_length: STFT hop length (samples).
        file_name: File name, used only for logging.
        count_output_path: Path to write the reconstructed audio to.
        audio_length_samples: Target length (samples) for the output audio.
        fft_freqs: Frequency (Hz) of each STFT bin.

    Returns:
        A tuple (audio_after_count, spectrum_magnitude_after_count,
        threshold_approved_frame_mask, bin_count_above_threshold_per_frame,
        peak_note_frame_count, plots):
            audio_after_count: reconstructed waveform after this filter.
            spectrum_magnitude_after_count: spectrum with rejected frames zeroed.
            threshold_approved_frame_mask: boolean mask, True for approved frames.
            bin_count_above_threshold_per_frame: number of active bins per frame.
            peak_note_frame_count: highest bin count observed among note frames.
            plots: list of 3 diagnostic plotting functions.
    """
    present_bin_median_threshold_factor = 0.7
    auto_approval_count_fraction = 0.5
    conditional_approval_count_fraction = 0.3
    entropy_mad_multiplier = 0.6

    bandpass_mask = (fft_freqs >= highpass_frequency) & (fft_freqs <= lowpass_frequency)
    bandpass_spectrum_magnitude = spectrum_magnitude_after_correlation_noise_filter[bandpass_mask, :]
    time_seconds = librosa.frames_to_time(np.arange(spectrum_magnitude_after_correlation_noise_filter.shape[1]), sr=sample_rate, hop_length=hop_length)
    total_frame_count = spectrum_magnitude_after_correlation_noise_filter.shape[1]

    median_magnitude_per_bin = np.median(bandpass_spectrum_magnitude, axis=1)
    magnitude_threshold_per_bin = present_bin_median_threshold_factor * median_magnitude_per_bin
    bin_count_above_threshold_per_frame = (bandpass_spectrum_magnitude > magnitude_threshold_per_bin[:, None]).sum(axis=0)
    note_frame_indices = np.concatenate([np.arange(note_start, note_end + 1) for note_start, note_end in notes]) if notes else np.array([], dtype=int)
    frame_belongs_to_note_mask = np.zeros(total_frame_count, dtype=bool)
    frame_belongs_to_note_mask[note_frame_indices] = True
    note_frame_counts = bin_count_above_threshold_per_frame[note_frame_indices]
    peak_note_frame_count = note_frame_counts.max()
    upper_count_threshold = auto_approval_count_fraction * peak_note_frame_count
    lower_count_threshold = conditional_approval_count_fraction * peak_note_frame_count

    bandpass_spectrum_power = bandpass_spectrum_magnitude ** 2
    entropy_energy_per_frame = bandpass_spectrum_power.sum(axis=0)
    entropy_frame_with_energy_mask = entropy_energy_per_frame > 0
    spectral_probability_per_frame = bandpass_spectrum_power[:, entropy_frame_with_energy_mask] / entropy_energy_per_frame[entropy_frame_with_energy_mask]
    entropy_per_frame = -(spectral_probability_per_frame * np.log(spectral_probability_per_frame + 1e-12)).sum(axis=0)
    max_entropy = np.log(bandpass_mask.sum())
    entropic_concentration_per_frame = 1 - entropy_per_frame / max_entropy
    median_entropic_concentration = np.median(entropic_concentration_per_frame)
    entropic_concentration_mad = np.median(np.abs(entropic_concentration_per_frame - median_entropic_concentration))
    entropic_concentration_threshold = median_entropic_concentration + entropy_mad_multiplier * entropic_concentration_mad

    high_count_approved_frame_mask = np.zeros(total_frame_count, dtype=bool)
    high_count_approved_frame_mask[note_frame_indices] = bin_count_above_threshold_per_frame[note_frame_indices] >= upper_count_threshold
    intermediate_range_frame_mask = np.zeros(total_frame_count, dtype=bool)
    intermediate_range_frame_mask[note_frame_indices] = (bin_count_above_threshold_per_frame[note_frame_indices] >= lower_count_threshold) & (bin_count_above_threshold_per_frame[note_frame_indices] < upper_count_threshold)
    conditional_entropy_approved_frame_mask = np.zeros(total_frame_count, dtype=bool)
    entropy_frame_indices_with_energy = np.flatnonzero(entropy_frame_with_energy_mask)
    conditional_entropy_approved_frame_mask[entropy_frame_indices_with_energy] = entropic_concentration_per_frame <= entropic_concentration_threshold

    threshold_approved_frame_mask = high_count_approved_frame_mask | (intermediate_range_frame_mask & conditional_entropy_approved_frame_mask)

    spectrum_magnitude_after_count = spectrum_magnitude_after_correlation_noise_filter * threshold_approved_frame_mask[None, :]
    audio_after_count = librosa.istft(spectrum_magnitude_after_count * np.exp(1j * phase_after_reduction), hop_length=hop_length, window="hann", center=True, length=audio_length_samples).astype(np.float32)
    sf.write(count_output_path, audio_after_count, sample_rate)
    print(f"  [{file_name}] audio after bin-count/entropy filter saved to {count_output_path} | {threshold_approved_frame_mask.sum()}/{note_frame_indices.size} frames approved")

    def plot_bin_count_per_frame(ax) -> None:
        ax.bar(time_seconds, bin_count_above_threshold_per_frame, width=hop_length / sample_rate, color="black", align="center")
        ax.axhline(upper_count_threshold, color="red", linewidth=1.0, linestyle="--")
        ax.axhline(lower_count_threshold, color="blue", linewidth=1.0, linestyle="--")
        ax.set_xlim(time_seconds[0], time_seconds[-1])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Number of bins with magnitude above threshold in the frame")
        ax.set_title(f"Bin count per frame ({threshold_approved_frame_mask.sum() / note_frame_indices.size * 100:.1f}% frames approved)")

    def plot_entropic_concentration_per_frame(ax) -> None:
        ax.bar(time_seconds[entropy_frame_with_energy_mask], entropic_concentration_per_frame, width=hop_length / sample_rate, color="black", align="center")
        ax.axhline(entropic_concentration_threshold, color="red", linewidth=1.0, linestyle="--")
        ax.set_xlim(time_seconds[0], time_seconds[-1])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spectral entropic concentration (1 - entropy/max entropy)")
        ax.set_title(f"Spectral entropic concentration per frame | threshold: {entropic_concentration_threshold:.3f}")

    def plot_spectrogram_after_count(ax) -> None:
        subsampling_step = max(1, spectrum_magnitude_after_count.shape[1] // 1000)
        max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude_after_count.shape[0] - 1)) + 1
        spectrum_magnitude_after_count_db = librosa.amplitude_to_db(spectrum_magnitude_after_count[:max_frequency_bin_index, ::subsampling_step], ref=spectrum_magnitude_after_correlation_noise_filter.max())
        zeroed_frame_percentage = (frame_belongs_to_note_mask & ~threshold_approved_frame_mask).mean() * 100
        ax.imshow(spectrum_magnitude_after_count_db, origin="lower", aspect="auto", extent=[0, spectrum_magnitude_after_count.shape[1] * hop_length / sample_rate, 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title(f"Spectrogram after bin-count/entropy filter (no expansion) | approved: {threshold_approved_frame_mask.mean() * 100:.1f}% | zeroed: {zeroed_frame_percentage:.1f}%")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    return audio_after_count, spectrum_magnitude_after_count, threshold_approved_frame_mask, bin_count_above_threshold_per_frame, peak_note_frame_count, [plot_bin_count_per_frame, plot_entropic_concentration_per_frame, plot_spectrogram_after_count]


def note_duration(spectrum_magnitude_after_correlation_noise_filter: np.ndarray, fft_freqs: np.ndarray, max_correlation_per_bin_note_expansion: np.ndarray, threshold_approved_frame_mask: np.ndarray, bin_count_above_threshold_per_frame: np.ndarray, peak_note_frame_count: float, hop_length: int, sample_rate: int, file_name: str, phase_after_reduction: np.ndarray, audio_length_samples: int, duration_output_path: str) -> tuple:
    """
    Prune short, weak "pure" notes and then re-expand the surviving notes'
    edges bin by bin based on how long each bin stays silent.

    Step 1 (pruning): group the threshold-approved frames into contiguous
    "pure" notes. A note is pruned (rejected) if its duration is below a
    MAD-based minimum threshold (median - |duration_mad_multiplier| * MAD,
    clipped at 0) AND its peak bin count is below
    min_pruning_count_fraction of peak_note_frame_count — i.e. it must be
    both short and weak to be pruned.

    Step 2 (expansion): starting from each surviving pure note's start/end
    frame, walk outward per bin (restricted to bins whose correlation from
    expand_notes exceeded bin_expansion_correlation_threshold). A bin keeps
    getting added to the note as long as it hasn't been silent (below
    relative_silence_threshold times that bin's own peak magnitude across
    the whole recording, not just within this note) for more than
    max_consecutive_silence_frames consecutive frames; once a bin exceeds
    that silence run length, it stops expanding, independently from the
    other bins.

    Args:
        spectrum_magnitude_after_correlation_noise_filter: STFT magnitude
            after filter_bins_by_correlation_and_probability.
        fft_freqs: Frequency (Hz) of each STFT bin.
        max_correlation_per_bin_note_expansion: Max correlation per band-pass
            bin, as returned by expand_notes.
        threshold_approved_frame_mask: Boolean frame mask from
            count_bins_per_frame_and_entropy.
        bin_count_above_threshold_per_frame: Per-frame active bin count, as
            returned by count_bins_per_frame_and_entropy.
        peak_note_frame_count: Peak in-note bin count, as returned by
            count_bins_per_frame_and_entropy.
        hop_length: STFT hop length (samples).
        sample_rate: Sample rate of the audio (Hz).
        file_name: File name, used only for logging.
        phase_after_reduction: Phase (radians) used for reconstruction.
        audio_length_samples: Target length (samples) for the output audio.
        duration_output_path: Path to write the reconstructed audio to.

    Returns:
        A tuple (audio_after_duration, spectrum_magnitude_after_duration,
        pixels_included_by_expansion_mask, plots):
            audio_after_duration: reconstructed waveform after this stage.
            spectrum_magnitude_after_duration: spectrum with the final
                per-bin/per-frame mask applied.
            pixels_included_by_expansion_mask: boolean array (same shape as
                the spectrum), True for time-frequency pixels added purely
                by the bin-by-bin silence expansion (not already approved).
            plots: list of 2 diagnostic plotting functions.
    """
    duration_mad_multiplier = -2
    min_pruning_count_fraction = 0.5
    bin_smoothing_window_size = 10
    bin_expansion_correlation_threshold = 0.75

    total_frame_count = spectrum_magnitude_after_correlation_noise_filter.shape[1]
    bandpass_mask = (fft_freqs >= highpass_frequency) & (fft_freqs <= lowpass_frequency)
    bandpass_spectrum_magnitude = spectrum_magnitude_after_correlation_noise_filter[bandpass_mask, :]
    smoothed_bandpass_spectrum_magnitude = uniform_filter1d(bandpass_spectrum_magnitude, size=bin_smoothing_window_size, axis=1, mode="nearest")
    correlation_approved_bin_mask = max_correlation_per_bin_note_expansion > bin_expansion_correlation_threshold

    threshold_approved_frame_indices = np.flatnonzero(threshold_approved_frame_mask)
    if threshold_approved_frame_indices.size > 0:
        breaks_between_approved_notes = np.where(np.diff(threshold_approved_frame_indices) > 1)[0]
        approved_note_starts = np.concatenate(([threshold_approved_frame_indices[0]], threshold_approved_frame_indices[breaks_between_approved_notes + 1]))
        approved_note_ends = np.concatenate((threshold_approved_frame_indices[breaks_between_approved_notes], [threshold_approved_frame_indices[-1]]))
        pure_notes_before_pruning = list(zip(approved_note_starts.tolist(), approved_note_ends.tolist()))
    else:
        pure_notes_before_pruning = []

    duration_seconds_per_pure_note_before_pruning = np.array([(note_end - note_start + 1) * hop_length / sample_rate for note_start, note_end in pure_notes_before_pruning])
    median_duration = np.median(duration_seconds_per_pure_note_before_pruning)
    duration_mad = np.median(np.abs(duration_seconds_per_pure_note_before_pruning - median_duration))
    min_duration_threshold = max(0, median_duration + duration_mad_multiplier * duration_mad)
    min_pruning_count_threshold = min_pruning_count_fraction * peak_note_frame_count

    max_count_per_pure_note_before_pruning = np.array([bin_count_above_threshold_per_frame[note_start:note_end + 1].max() for note_start, note_end in pure_notes_before_pruning])
    pruned_note_mask = (duration_seconds_per_pure_note_before_pruning < min_duration_threshold) & (max_count_per_pure_note_before_pruning < min_pruning_count_threshold)

    threshold_approved_frame_mask = threshold_approved_frame_mask.copy()
    for (pure_note_start, pure_note_end), note_pruned in zip(pure_notes_before_pruning, pruned_note_mask):
        if note_pruned:
            threshold_approved_frame_mask[pure_note_start:pure_note_end + 1] = False
    pure_notes = [pure_note for pure_note, note_pruned in zip(pure_notes_before_pruning, pruned_note_mask) if not note_pruned]
    pruning_note_colors = np.where(pruned_note_mask, "red", "black")

    relative_silence_threshold = 0.05
    max_consecutive_silence_frames = 5
    peak_magnitude_per_bin = smoothed_bandpass_spectrum_magnitude.max(axis=1)
    silence_threshold_per_bin = relative_silence_threshold * peak_magnitude_per_bin
    expanded_approved_bin_frame_mask = np.broadcast_to(threshold_approved_frame_mask, spectrum_magnitude_after_correlation_noise_filter.shape).copy()
    bandpass_bin_indices = np.flatnonzero(bandpass_mask)
    for pure_note_start, pure_note_end in pure_notes:
        for edge_index, direction in ((pure_note_start, -1), (pure_note_end, 1)):
            active_bins_mask = correlation_approved_bin_mask.copy()
            silence_frame_count_per_bin = np.zeros(smoothed_bandpass_spectrum_magnitude.shape[0], dtype=int)
            neighbor_frame_index = edge_index + direction
            while 0 <= neighbor_frame_index < total_frame_count and active_bins_mask.any():
                neighbor_magnitude_per_bin = smoothed_bandpass_spectrum_magnitude[:, neighbor_frame_index]
                bins_with_signal_mask = neighbor_magnitude_per_bin > silence_threshold_per_bin
                silence_frame_count_per_bin[active_bins_mask & bins_with_signal_mask] = 0
                silence_frame_count_per_bin[active_bins_mask & ~bins_with_signal_mask] += 1
                active_bins_mask &= silence_frame_count_per_bin < max_consecutive_silence_frames
                if not active_bins_mask.any():
                    break
                expanded_approved_bin_frame_mask[bandpass_bin_indices[active_bins_mask], neighbor_frame_index] = True
                neighbor_frame_index += direction

    expanded_approved_frame_mask = expanded_approved_bin_frame_mask[bandpass_mask, :].any(axis=0)
    pixels_included_by_expansion_mask = expanded_approved_bin_frame_mask & ~np.broadcast_to(threshold_approved_frame_mask, spectrum_magnitude_after_correlation_noise_filter.shape)

    spectrum_magnitude_after_duration = spectrum_magnitude_after_correlation_noise_filter * expanded_approved_bin_frame_mask
    audio_after_duration = librosa.istft(spectrum_magnitude_after_duration * np.exp(1j * phase_after_reduction), hop_length=hop_length, window="hann", center=True, length=audio_length_samples).astype(np.float32)
    sf.write(duration_output_path, audio_after_duration, sample_rate)
    print(f"  [{file_name}] audio after note expansion saved to {duration_output_path} | {len(pure_notes)}/{len(pure_notes_before_pruning)} notes kept after pruning")

    def plot_min_duration_pruning(ax) -> None:
        pre_pruning_note_start_times = np.array([note_start * hop_length / sample_rate for note_start, note_end in pure_notes_before_pruning])
        ax.bar(pre_pruning_note_start_times, duration_seconds_per_pure_note_before_pruning, width=duration_seconds_per_pure_note_before_pruning, align="edge", color=pruning_note_colors)
        ax.axhline(min_duration_threshold, color="green", linewidth=1.0, linestyle="--")
        ax.set_xlim(0, total_frame_count * hop_length / sample_rate)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pure note duration (s)")
        ax.set_title(f"Minimum-duration pruning ({min_duration_threshold:.2f}s) AND peak count < {min_pruning_count_threshold:.1f} bins | {pruned_note_mask.sum()}/{len(pure_notes_before_pruning)} notes pruned")
        ax.legend(handles=[matplotlib.patches.Patch(color="black", label="Note kept"), matplotlib.patches.Patch(color="red", label="Note pruned")], loc="upper right", fontsize=8)

    def plot_spectrogram_after_duration_expansion(ax) -> None:
        subsampling_step = max(1, spectrum_magnitude_after_duration.shape[1] // 1000)
        max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude_after_duration.shape[0] - 1)) + 1
        spectrum_magnitude_after_duration_db = librosa.amplitude_to_db(spectrum_magnitude_after_duration[:max_frequency_bin_index, ::subsampling_step], ref=spectrum_magnitude_after_correlation_noise_filter.max())
        ax.imshow(spectrum_magnitude_after_duration_db, origin="lower", aspect="auto", extent=[0, spectrum_magnitude_after_duration.shape[1] * hop_length / sample_rate, 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title(f"Spectrogram after expansion (bins with correlation > {bin_expansion_correlation_threshold}) | approved: {expanded_approved_frame_mask.mean() * 100:.1f}%")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    return audio_after_duration, spectrum_magnitude_after_duration, pixels_included_by_expansion_mask, [plot_min_duration_pruning, plot_spectrogram_after_duration_expansion]


def centroid_analysis(spectrum_magnitude_after_correlation_noise_filter: np.ndarray, spectrum_magnitude_after_duration: np.ndarray, fft_freqs: np.ndarray, hop_length: int, sample_rate: int, pixels_included_by_expansion_mask: np.ndarray, file_name: str, phase_after_reduction: np.ndarray, audio_length_samples: int, centroid_output_path: str) -> tuple:
    """
    Validate the time-frequency pixels added purely by note_duration's
    bin-by-bin silence expansion, using the frame's spectral centroid as a
    consistency check, and zero out any frame that fails.

    A frame is approved if either:
      - its spectral centroid falls within a MAD-based range around the
        median centroid of all energy-containing frames (median +/-
        multiplier * MAD, with separate lower/upper multipliers); or
      - its centroid falls into a histogram bin (over all frame centroids)
        whose normalized count exceeds normalized_centroid_count_approval_threshold,
        or into a bin within max_centroid_count_approval_distance Hz of one
        that does.
    Any frame that was included only by the earlier silence expansion (i.e.
    it wasn't already part of the base note) and fails this centroid check
    gets fully zeroed out.

    Args:
        spectrum_magnitude_after_correlation_noise_filter: STFT magnitude
            after filter_bins_by_correlation_and_probability, used only to
            determine bandpass bins with any energy (for histogram range).
        spectrum_magnitude_after_duration: STFT magnitude after note_duration.
        fft_freqs: Frequency (Hz) of each STFT bin.
        hop_length: STFT hop length (samples).
        sample_rate: Sample rate of the audio (Hz).
        pixels_included_by_expansion_mask: Boolean array from note_duration,
            True for pixels added purely by its silence-based expansion.
        file_name: File name, used only for logging.
        phase_after_reduction: Phase (radians) used for reconstruction.
        audio_length_samples: Target length (samples) for the output audio.
        centroid_output_path: Path to write the reconstructed audio to.

    Returns:
        A tuple (audio_after_centroid, spectrum_magnitude_after_centroid,
        frame_rejected_by_expansion_mask, plots):
            audio_after_centroid: reconstructed waveform after this stage.
            spectrum_magnitude_after_centroid: spectrum with rejected frames
                (columns) zeroed.
            frame_rejected_by_expansion_mask: boolean mask, True for frames
                that were expansion-only and failed the centroid check.
            plots: list of 4 diagnostic plotting functions.
    """
    normalized_centroid_count_approval_threshold = 0.5
    max_centroid_count_approval_distance = 200
    lower_centroid_mad_multiplier = 3
    upper_centroid_mad_multiplier = 3

    total_frame_count = spectrum_magnitude_after_correlation_noise_filter.shape[1]
    bandpass_mask = (fft_freqs >= highpass_frequency) & (fft_freqs <= lowpass_frequency)
    bandpass_fft_freqs = fft_freqs[bandpass_mask]
    bandpass_spectrum_magnitude = spectrum_magnitude_after_correlation_noise_filter[bandpass_mask, :]
    bandpass_spectrum_magnitude_after_duration = spectrum_magnitude_after_duration[bandpass_mask, :]
    time_seconds = librosa.frames_to_time(np.arange(total_frame_count), sr=sample_rate, hop_length=hop_length)

    centroid_energy_per_frame = bandpass_spectrum_magnitude_after_duration.sum(axis=0)
    centroid_frame_with_energy_mask = centroid_energy_per_frame > 0
    centroid_per_frame = np.zeros(total_frame_count, dtype=np.float64)
    centroid_per_frame[centroid_frame_with_energy_mask] = (bandpass_fft_freqs[:, None] * bandpass_spectrum_magnitude_after_duration[:, centroid_frame_with_energy_mask]).sum(axis=0) / centroid_energy_per_frame[centroid_frame_with_energy_mask]

    bandpass_bins_with_energy_mask = bandpass_spectrum_magnitude.max(axis=1) > 0
    min_frequency_with_energy = bandpass_fft_freqs[bandpass_bins_with_energy_mask].min()
    max_frequency_with_energy = bandpass_fft_freqs[bandpass_bins_with_energy_mask].max()
    histogram_lower_limit = min_frequency_with_energy - 50
    histogram_upper_limit = max_frequency_with_energy + 50
    fft_frequency_resolution = bandpass_fft_freqs[1] - bandpass_fft_freqs[0]
    histogram_bin_count = int(round((histogram_upper_limit - histogram_lower_limit) / fft_frequency_resolution))
    histogram_counts, histogram_edges = np.histogram(centroid_per_frame[centroid_frame_with_energy_mask], bins=histogram_bin_count, range=(histogram_lower_limit, histogram_upper_limit))
    normalized_count_per_bin = histogram_counts / histogram_counts.max()
    histogram_bin_centers = (histogram_edges[:-1] + histogram_edges[1:]) / 2
    histogram_bin_width = histogram_edges[1] - histogram_edges[0]

    count_approved_bin_mask = (normalized_count_per_bin > normalized_centroid_count_approval_threshold)
    count_approved_bin_frequencies = histogram_bin_centers[count_approved_bin_mask]
    nearest_approved_bin_distance = np.abs(histogram_bin_centers[:, None] - count_approved_bin_frequencies[None, :]).min(axis=1) if count_approved_bin_frequencies.size > 0 else np.full(histogram_bin_centers.size, np.inf)
    count_approved_bin_mask |= nearest_approved_bin_distance < max_centroid_count_approval_distance
    histogram_bin_index_per_frame = np.clip(np.floor((centroid_per_frame - histogram_lower_limit) / histogram_bin_width).astype(int), 0, histogram_bin_count - 1)
    count_bin_approved_frame_mask = count_approved_bin_mask[histogram_bin_index_per_frame]

    median_centroid_per_frame = np.median(centroid_per_frame[centroid_frame_with_energy_mask])
    centroid_mad_per_frame = np.median(np.abs(centroid_per_frame[centroid_frame_with_energy_mask] - median_centroid_per_frame))
    lower_centroid_threshold = median_centroid_per_frame - lower_centroid_mad_multiplier * centroid_mad_per_frame
    upper_centroid_threshold = median_centroid_per_frame + upper_centroid_mad_multiplier * centroid_mad_per_frame
    frame_within_centroid_mad_mask = (centroid_per_frame >= lower_centroid_threshold) & (centroid_per_frame <= upper_centroid_threshold)
    centroid_approved_frame_mask = frame_within_centroid_mad_mask | count_bin_approved_frame_mask
    frame_included_only_by_expansion_mask = pixels_included_by_expansion_mask.any(axis=0)
    frame_rejected_by_expansion_mask = frame_included_only_by_expansion_mask & ~centroid_approved_frame_mask

    spectrum_magnitude_expanded_pixels = spectrum_magnitude_after_duration * pixels_included_by_expansion_mask
    spectrum_magnitude_after_centroid = spectrum_magnitude_after_duration.copy()
    spectrum_magnitude_after_centroid[:, frame_rejected_by_expansion_mask] = 0

    audio_after_centroid = librosa.istft(spectrum_magnitude_after_centroid * np.exp(1j * phase_after_reduction), hop_length=hop_length, window="hann", center=True, length=audio_length_samples).astype(np.float32)
    sf.write(centroid_output_path, audio_after_centroid, sample_rate)
    print(f"  [{file_name}] audio after centroid-based frame removal saved to {centroid_output_path} | {frame_rejected_by_expansion_mask.sum()} frames zeroed")

    def plot_spectrogram_expanded_pixels(ax) -> None:
        subsampling_step = max(1, spectrum_magnitude_expanded_pixels.shape[1] // 1000)
        max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude_expanded_pixels.shape[0] - 1)) + 1
        spectrum_magnitude_expanded_pixels_db = librosa.amplitude_to_db(spectrum_magnitude_expanded_pixels[:max_frequency_bin_index, ::subsampling_step], ref=spectrum_magnitude_after_duration.max())
        ax.imshow(spectrum_magnitude_expanded_pixels_db, origin="lower", aspect="auto", extent=[0, spectrum_magnitude_expanded_pixels.shape[1] * hop_length / sample_rate, 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title(f"Pixels included by note expansion | {pixels_included_by_expansion_mask.sum()} pixels")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    def plot_centroid_histogram(ax) -> None:
        histogram_bin_colors = np.where(count_approved_bin_mask, "black", "red")
        ax.bar(histogram_bin_centers, histogram_counts, width=fft_frequency_resolution, color=histogram_bin_colors, align="center")
        ax.axhline(normalized_centroid_count_approval_threshold * histogram_counts.max(), color="green", linewidth=1.0, linestyle="--")
        ax.set_xlim(histogram_lower_limit, histogram_upper_limit)
        ax.set_xlabel("Centroid frequency (Hz)")
        ax.set_ylabel("Number of frames")
        ax.set_title(f"Histogram of the centroid frequency per frame | approved bins: {count_approved_bin_mask.sum()}/{histogram_bin_count}")
        ax.legend(handles=[matplotlib.patches.Patch(color="black", label="Approved bin"), matplotlib.patches.Patch(color="red", label="Not approved bin")], loc="upper right", fontsize=8)

    def plot_centroid_per_frame(ax) -> None:
        centroid_bar_colors = np.where(frame_rejected_by_expansion_mask, "red", "black")
        ax.bar(time_seconds, centroid_per_frame, width=hop_length / sample_rate, color=centroid_bar_colors, align="center")
        ax.axhline(lower_centroid_threshold, color="green", linewidth=1.0, linestyle="--")
        ax.axhline(upper_centroid_threshold, color="green", linewidth=1.0, linestyle="--")
        ax.set_xlim(time_seconds[0], time_seconds[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spectral centroid (Hz)")
        ax.set_title(f"Spectral centroid per frame | {frame_rejected_by_expansion_mask.sum()} expansion frames rejected")

    def plot_spectrogram_after_centroid(ax) -> None:
        subsampling_step = max(1, spectrum_magnitude_after_centroid.shape[1] // 1000)
        max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude_after_centroid.shape[0] - 1)) + 1
        spectrum_magnitude_after_centroid_db = librosa.amplitude_to_db(spectrum_magnitude_after_centroid[:max_frequency_bin_index, ::subsampling_step], ref=spectrum_magnitude_after_duration.max())
        ax.imshow(spectrum_magnitude_after_centroid_db, origin="lower", aspect="auto", extent=[0, spectrum_magnitude_after_centroid.shape[1] * hop_length / sample_rate, 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title(f"Spectrogram after removing frames rejected by centroid | {frame_rejected_by_expansion_mask.sum()} frames zeroed")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    return audio_after_centroid, spectrum_magnitude_after_centroid, frame_rejected_by_expansion_mask, [plot_spectrogram_expanded_pixels, plot_centroid_histogram, plot_centroid_per_frame, plot_spectrogram_after_centroid]


def mean_continuous_band_length(spectrum_magnitude_after_centroid: np.ndarray, spectrum_magnitude_after_correlation_noise_filter: np.ndarray, pixels_included_by_expansion_mask: np.ndarray, threshold_approved_frame_mask: np.ndarray, hop_length: int, sample_rate: int, file_name: str, phase_after_reduction: np.ndarray, audio_length_samples: int, mean_length_output_path: str) -> tuple:
    """
    Reject expansion-only frames whose active bins form short, thin
    frequency bands rather than one (or few) wide continuous bands, as a
    proxy for "this looks like scattered noise, not part of a call".

    For each frame, the active-bin mask (magnitude > 0, or included by the
    earlier note expansion) is dilated by 1 bin along frequency, then split
    into contiguous bands; the number of bands and the total active-bin
    count give an average band length per frame (active bins / band count).
    A frame is rejected if all of the following hold:
      - its mean band length is below mean_length_percentage_threshold of
        the peak mean band length across the recording,
      - its normalized max magnitude is below normalized_magnitude_threshold,
      - it was included only by the earlier note expansion (not part of the
        original approved note), and
      - it is not itself a threshold-approved frame.

    Args:
        spectrum_magnitude_after_centroid: STFT magnitude after
            centroid_analysis.
        spectrum_magnitude_after_correlation_noise_filter: STFT magnitude
            after filter_bins_by_correlation_and_probability, used only for
            the normalized max magnitude reference.
        pixels_included_by_expansion_mask: Boolean array from note_duration.
        threshold_approved_frame_mask: Boolean frame mask from
            count_bins_per_frame_and_entropy.
        hop_length: STFT hop length (samples).
        sample_rate: Sample rate of the audio (Hz).
        file_name: File name, used only for logging.
        phase_after_reduction: Phase (radians) used for reconstruction.
        audio_length_samples: Target length (samples) for the output audio.
        mean_length_output_path: Path to write the reconstructed audio to.

    Returns:
        A tuple (audio_after_mean_length, spectrum_magnitude_after_mean_length,
        frame_rejected_by_mean_length_mask, plots):
            audio_after_mean_length: reconstructed waveform after this stage.
            spectrum_magnitude_after_mean_length: spectrum with rejected
                frames (columns) zeroed.
            frame_rejected_by_mean_length_mask: boolean mask, True for
                rejected frames.
            plots: list of 2 diagnostic plotting functions.
    """
    mean_length_percentage_threshold = 0.02
    normalized_magnitude_threshold = 0.15

    time_seconds = librosa.frames_to_time(np.arange(spectrum_magnitude_after_centroid.shape[1]), sr=sample_rate, hop_length=hop_length)
    active_bins_mask = (spectrum_magnitude_after_centroid > 0) | pixels_included_by_expansion_mask
    dilated_mask = active_bins_mask.copy()
    dilated_mask[:-1] |= active_bins_mask[1:]
    dilated_mask[1:] |= active_bins_mask[:-1]
    band_start_mask = dilated_mask.copy()
    band_start_mask[1:] &= ~dilated_mask[:-1]
    continuous_band_count_per_frame = band_start_mask.sum(axis=0)
    active_bins_per_frame = active_bins_mask.sum(axis=0)
    mean_band_length_per_frame = np.divide(active_bins_per_frame, continuous_band_count_per_frame, out=np.zeros_like(active_bins_per_frame, dtype=float), where=continuous_band_count_per_frame > 0)
    mean_length_threshold = mean_length_percentage_threshold * mean_band_length_per_frame.max()
    normalized_max_magnitude_per_frame = spectrum_magnitude_after_correlation_noise_filter.max(axis=0) / (spectrum_magnitude_after_correlation_noise_filter.max() + 1e-12)
    frame_included_by_expansion_mask = pixels_included_by_expansion_mask.any(axis=0)
    frame_rejected_by_mean_length_mask = (mean_band_length_per_frame < mean_length_threshold) & (normalized_max_magnitude_per_frame < normalized_magnitude_threshold) & frame_included_by_expansion_mask & ~threshold_approved_frame_mask
    bar_colors = np.where(threshold_approved_frame_mask, "black", np.where(frame_included_by_expansion_mask, "red", "gray"))

    spectrum_magnitude_after_mean_length = spectrum_magnitude_after_centroid.copy()
    spectrum_magnitude_after_mean_length[:, frame_rejected_by_mean_length_mask] = 0

    audio_after_mean_length = librosa.istft(spectrum_magnitude_after_mean_length * np.exp(1j * phase_after_reduction), hop_length=hop_length, window="hann", center=True, length=audio_length_samples).astype(np.float32)
    sf.write(mean_length_output_path, audio_after_mean_length, sample_rate)
    print(f"  [{file_name}] audio after mean-length frame removal saved to {mean_length_output_path} | {frame_rejected_by_mean_length_mask.sum()} frames zeroed")

    def plot_mean_continuous_band_length(ax) -> None:
        ax.bar(time_seconds, mean_band_length_per_frame, width=hop_length / sample_rate, color=bar_colors, align="center")
        ax.axhline(mean_length_threshold, color="blue", linestyle="--", linewidth=1)
        ax.set_xlim(time_seconds[0], time_seconds[-1])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean length of continuous bin bands (bins)")
        ax.set_title(f"Mean continuous band length per frame, post-expansion | {frame_rejected_by_mean_length_mask.sum()} frames rejected")
        ax.legend(handles=[matplotlib.patches.Patch(color="black", label="Original note (pre-expansion)"), matplotlib.patches.Patch(color="red", label="Expanded"), matplotlib.patches.Patch(color="gray", label="Other"), matplotlib.lines.Line2D([0], [0], color="blue", linestyle="--", label=f"Threshold ({mean_length_percentage_threshold:.0%} of peak)")], loc="upper right", fontsize=8)

    def plot_spectrogram_after_mean_length(ax) -> None:
        subsampling_step = max(1, spectrum_magnitude_after_mean_length.shape[1] // 1000)
        max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude_after_mean_length.shape[0] - 1)) + 1
        spectrum_magnitude_after_mean_length_db = librosa.amplitude_to_db(spectrum_magnitude_after_mean_length[:max_frequency_bin_index, ::subsampling_step], ref=spectrum_magnitude_after_mean_length.max())
        ax.imshow(spectrum_magnitude_after_mean_length_db, origin="lower", aspect="auto", extent=[0, spectrum_magnitude_after_mean_length.shape[1] * hop_length / sample_rate, 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title("Spectrogram after mean-length filter")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    return audio_after_mean_length, spectrum_magnitude_after_mean_length, frame_rejected_by_mean_length_mask, [plot_mean_continuous_band_length, plot_spectrogram_after_mean_length]


def compute_time_bounds_of_continuous_energy_frames(spectrum_magnitude: np.ndarray, hop_length: int, sample_rate: int) -> tuple:
    """
    Find every contiguous run of frames that has any non-zero energy
    (summed across all bins) and return their start/end frame indices and
    the corresponding start/end times in seconds.

    Args:
        spectrum_magnitude: STFT magnitude, shape (n_freq_bins, n_frames).
        hop_length: STFT hop length (samples).
        sample_rate: Sample rate of the audio (Hz).

    Returns:
        A tuple (note_starts, note_ends, note_start_times, note_end_times):
            note_starts: start frame index of each contiguous run.
            note_ends: end frame index (inclusive) of each contiguous run.
            note_start_times: note_starts converted to seconds.
            note_end_times: (note_ends + 1) converted to seconds.
    """
    frame_with_energy_mask = (spectrum_magnitude > 0).any(axis=0)
    frame_with_energy_indices = np.flatnonzero(frame_with_energy_mask)
    if frame_with_energy_indices.size > 0:
        breaks_between_notes = np.where(np.diff(frame_with_energy_indices) > 1)[0]
        note_starts = np.concatenate(([frame_with_energy_indices[0]], frame_with_energy_indices[breaks_between_notes + 1]))
        note_ends = np.concatenate((frame_with_energy_indices[breaks_between_notes], [frame_with_energy_indices[-1]]))
    else:
        note_starts = np.array([], dtype=int)
        note_ends = np.array([], dtype=int)
    note_start_times = note_starts * hop_length / sample_rate
    note_end_times = (note_ends + 1) * hop_length / sample_rate
    return note_starts, note_ends, note_start_times, note_end_times


def deexpand_erroneously_long_notes(spectrum_magnitude_after_mean_length: np.ndarray, approved_frame_mask: np.ndarray, hop_length: int, sample_rate: int, full_fft_freqs: np.ndarray, dominant_band_min_frequencies: float, dominant_band_max_frequencies: float, file_name: str, phase_after_reduction: np.ndarray, audio_length_samples: int, deexpansion_output_path: str) -> tuple:
    """
    Shrink ("de-expand") surviving notes whose duration grew
    disproportionately relative to their original call-detection (VAD)
    duration, which usually signals that earlier expansion stages
    over-extended a note into noise, unless the note's edges are strongly
    loud (in which case the long duration is trusted and left alone).

    For each surviving note (a contiguous run of energy in
    spectrum_magnitude_after_mean_length):
      1. Compute duration_after_mean_length / duration_after_vad (the ratio
         of how long the note currently is vs. how long it was right after
         call detection, before any expansion).
      2. Get robust (median + k*MAD) thresholds for that ratio (one regular,
         one "extreme") and for the absolute post-mean-length duration.
      3. Flag a note for de-expansion if EITHER: its ratio and its duration
         both exceed their regular thresholds, OR its ratio exceeds the
         extreme threshold while its peak magnitude stays below
         extreme_note_max_magnitude_threshold (very long relative to VAD,
         but too weak to trust).
      4. Un-flag any note whose start and end edges (first/last ~10% of its
         frames, within the dominant band) are both louder than a
         MAD-based strong-edge threshold - a strongly bounded loud note is
         trusted even if it is comparatively long.
      5. For each flagged note, repeatedly zero out whichever endpoint frame
         (start or end) has less total energy, shrinking the note one frame
         at a time until its duration drops to the de-expansion duration
         threshold (median + deexpansion_duration_threshold_mad_multiplier*MAD,
         a tighter, closer-to-median target than the flagging threshold in
         step 2) or it collapses to a single frame.

    Args:
        spectrum_magnitude_after_mean_length: STFT magnitude after
            mean_continuous_band_length.
        approved_frame_mask: Boolean frame mask from detect_call_frames
            (pre-expansion), used as the "VAD" duration reference.
        hop_length: STFT hop length (samples).
        sample_rate: Sample rate of the audio (Hz).
        full_fft_freqs: Frequency (Hz) of every STFT bin (full spectrum,
            not just the band-pass range).
        dominant_band_min_frequencies: Lower edge(s) (Hz) of the dominant band(s).
        dominant_band_max_frequencies: Upper edge(s) (Hz) of the dominant band(s).
        file_name: File name, used only for logging.
        phase_after_reduction: Phase (radians) used for reconstruction.
        audio_length_samples: Target length (samples) for the output audio.
        deexpansion_output_path: Path to write the reconstructed audio to.

    Returns:
        A tuple (audio_after_deexpansion, spectrum_magnitude_after_deexpansion,
        note_flagged_for_deexpansion_mask, plots):
            audio_after_deexpansion: reconstructed waveform after this stage.
            spectrum_magnitude_after_deexpansion: spectrum after shrinking
                flagged notes.
            note_flagged_for_deexpansion_mask: boolean mask, True for
                surviving notes that were shrunk.
            plots: list of 4 diagnostic plotting functions.
    """
    ratio_mad_multiplier = 1
    extreme_ratio_mad_multiplier = 7
    duration_mad_multiplier = 2
    deexpansion_duration_threshold_mad_multiplier = 0.25
    strong_edge_mad_multiplier = 1
    extreme_note_max_magnitude_threshold = 0.1

    surviving_note_starts_mean_length, surviving_note_ends_mean_length, surviving_note_start_times_mean_length, surviving_note_end_times_mean_length = compute_time_bounds_of_continuous_energy_frames(spectrum_magnitude_after_mean_length, hop_length, sample_rate)

    mean_length_duration_seconds_per_surviving_note = (surviving_note_ends_mean_length - surviving_note_starts_mean_length + 1) * hop_length / sample_rate
    vad_duration_seconds_per_surviving_note = np.array([approved_frame_mask[note_start:note_end + 1].sum() for note_start, note_end in zip(surviving_note_starts_mean_length, surviving_note_ends_mean_length)]) * hop_length / sample_rate
    mean_length_to_vad_duration_ratio_per_surviving_note = np.divide(mean_length_duration_seconds_per_surviving_note, vad_duration_seconds_per_surviving_note, out=np.zeros_like(mean_length_duration_seconds_per_surviving_note), where=vad_duration_seconds_per_surviving_note > 0)
    max_magnitude_per_surviving_note = np.array([spectrum_magnitude_after_mean_length[:, note_start:note_end + 1].max() for note_start, note_end in zip(surviving_note_starts_mean_length, surviving_note_ends_mean_length)])

    note_with_positive_vad_duration_mask = vad_duration_seconds_per_surviving_note > 0
    print(f"  [{file_name}] total notes: {note_with_positive_vad_duration_mask.size}, real notes: {note_with_positive_vad_duration_mask.sum()}")

    median_mean_length_vad_duration_ratio = np.median(mean_length_to_vad_duration_ratio_per_surviving_note[note_with_positive_vad_duration_mask])
    mean_length_vad_duration_ratio_mad = np.median(np.abs(mean_length_to_vad_duration_ratio_per_surviving_note[note_with_positive_vad_duration_mask] - median_mean_length_vad_duration_ratio))
    mean_length_vad_duration_ratio_threshold = median_mean_length_vad_duration_ratio + ratio_mad_multiplier * mean_length_vad_duration_ratio_mad
    extreme_mean_length_vad_duration_ratio_threshold = median_mean_length_vad_duration_ratio + extreme_ratio_mad_multiplier * mean_length_vad_duration_ratio_mad

    median_mean_length_duration_per_surviving_note = np.median(mean_length_duration_seconds_per_surviving_note[note_with_positive_vad_duration_mask])
    mean_length_duration_mad_per_surviving_note = np.median(np.abs(mean_length_duration_seconds_per_surviving_note[note_with_positive_vad_duration_mask] - median_mean_length_duration_per_surviving_note))
    mean_length_duration_threshold_per_surviving_note = median_mean_length_duration_per_surviving_note + duration_mad_multiplier * mean_length_duration_mad_per_surviving_note
    deexpansion_mean_length_duration_threshold_per_surviving_note = median_mean_length_duration_per_surviving_note + deexpansion_duration_threshold_mad_multiplier * mean_length_duration_mad_per_surviving_note

    note_flagged_by_ratio_mask = mean_length_to_vad_duration_ratio_per_surviving_note > mean_length_vad_duration_ratio_threshold
    note_flagged_by_duration_mask = mean_length_duration_seconds_per_surviving_note > mean_length_duration_threshold_per_surviving_note
    note_extreme_ratio_above_threshold_mask = mean_length_to_vad_duration_ratio_per_surviving_note > extreme_mean_length_vad_duration_ratio_threshold
    note_valid_extreme_ratio_mask = note_extreme_ratio_above_threshold_mask & (max_magnitude_per_surviving_note < extreme_note_max_magnitude_threshold)
    note_invalid_extreme_ratio_mask = note_extreme_ratio_above_threshold_mask & (max_magnitude_per_surviving_note >= extreme_note_max_magnitude_threshold)
    note_flagged_for_deexpansion_mask = (note_flagged_by_ratio_mask & note_flagged_by_duration_mask) | note_valid_extreme_ratio_mask

    dominant_band_bins_mask = np.zeros_like(full_fft_freqs, dtype=bool)
    for band_min_frequency, band_max_frequency in zip(dominant_band_min_frequencies, dominant_band_max_frequencies):
        dominant_band_bins_mask |= (full_fft_freqs >= band_min_frequency) & (full_fft_freqs <= band_max_frequency)
    all_surviving_note_frame_indices = np.concatenate([np.arange(note_start, note_end + 1) for note_start, note_end in zip(surviving_note_starts_mean_length, surviving_note_ends_mean_length)]) if surviving_note_starts_mean_length.size > 0 else np.array([], dtype=int)
    dominant_band_magnitude_all_surviving_notes = spectrum_magnitude_after_mean_length[np.ix_(dominant_band_bins_mask, all_surviving_note_frame_indices)]
    median_dominant_band_magnitude = np.median(dominant_band_magnitude_all_surviving_notes)
    dominant_band_magnitude_mad = np.median(np.abs(dominant_band_magnitude_all_surviving_notes - median_dominant_band_magnitude))
    strong_edge_magnitude_threshold = median_dominant_band_magnitude - strong_edge_mad_multiplier * dominant_band_magnitude_mad

    surviving_note_lengths_mean_length = surviving_note_ends_mean_length - surviving_note_starts_mean_length + 1
    edge_frame_count_per_surviving_note = np.ceil(surviving_note_lengths_mean_length * 0.1).astype(int)
    median_start_edge_magnitude_per_surviving_note = np.array([np.median(spectrum_magnitude_after_mean_length[np.ix_(dominant_band_bins_mask, np.arange(note_start, note_start + edge_frame_count))]) for note_start, edge_frame_count in zip(surviving_note_starts_mean_length, edge_frame_count_per_surviving_note)])
    median_end_edge_magnitude_per_surviving_note = np.array([np.median(spectrum_magnitude_after_mean_length[np.ix_(dominant_band_bins_mask, np.arange(note_end - edge_frame_count + 1, note_end + 1))]) for note_end, edge_frame_count in zip(surviving_note_ends_mean_length, edge_frame_count_per_surviving_note)])
    strong_edge_note_mask = (median_start_edge_magnitude_per_surviving_note > strong_edge_magnitude_threshold) & (median_end_edge_magnitude_per_surviving_note > strong_edge_magnitude_threshold)
    note_flagged_for_deexpansion_mask = note_flagged_for_deexpansion_mask & ~strong_edge_note_mask

    spectrum_magnitude_after_deexpansion = spectrum_magnitude_after_mean_length.copy()
    for flagged_note_index in np.flatnonzero(note_flagged_for_deexpansion_mask):
        deexpanded_note_start_frame = surviving_note_starts_mean_length[flagged_note_index]
        deexpanded_note_end_frame = surviving_note_ends_mean_length[flagged_note_index]
        deexpanded_note_duration_seconds = (deexpanded_note_end_frame - deexpanded_note_start_frame + 1) * hop_length / sample_rate
        while deexpanded_note_duration_seconds > deexpansion_mean_length_duration_threshold_per_surviving_note and deexpanded_note_end_frame > deexpanded_note_start_frame:
            deexpanded_note_start_frame_energy = np.sum(spectrum_magnitude_after_deexpansion[:, deexpanded_note_start_frame] ** 2)
            deexpanded_note_end_frame_energy = np.sum(spectrum_magnitude_after_deexpansion[:, deexpanded_note_end_frame] ** 2)
            if deexpanded_note_start_frame_energy <= deexpanded_note_end_frame_energy:
                spectrum_magnitude_after_deexpansion[:, deexpanded_note_start_frame] = 0
                deexpanded_note_start_frame += 1
            else:
                spectrum_magnitude_after_deexpansion[:, deexpanded_note_end_frame] = 0
                deexpanded_note_end_frame -= 1
            deexpanded_note_duration_seconds = (deexpanded_note_end_frame - deexpanded_note_start_frame + 1) * hop_length / sample_rate

    audio_after_deexpansion = librosa.istft(spectrum_magnitude_after_deexpansion * np.exp(1j * phase_after_reduction), hop_length=hop_length, window="hann", center=True, length=audio_length_samples).astype(np.float32)
    sf.write(deexpansion_output_path, audio_after_deexpansion, sample_rate)
    print(f"  [{file_name}] audio after note de-expansion saved to {deexpansion_output_path} | {note_flagged_for_deexpansion_mask.sum()}/{note_flagged_for_deexpansion_mask.size} notes de-expanded")

    def plot_mean_length_vad_duration_ratio(ax) -> None:
        ratio_bar_colors = np.where(note_invalid_extreme_ratio_mask, "blue", np.where(note_flagged_by_ratio_mask, "red", "black"))
        ax.bar(surviving_note_start_times_mean_length, mean_length_to_vad_duration_ratio_per_surviving_note, width=mean_length_duration_seconds_per_surviving_note, align="edge", color=ratio_bar_colors)
        ax.axhline(mean_length_vad_duration_ratio_threshold, color="red", linewidth=1.0, linestyle="--")
        ax.axhline(extreme_mean_length_vad_duration_ratio_threshold, color="darkred", linewidth=1.0, linestyle=":")
        ax.set_xlim(0, spectrum_magnitude_after_mean_length.shape[1] * hop_length / sample_rate)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Ratio of post-mean-length duration / post-VAD duration")
        ax.set_title("Ratio between post-mean-length duration and post-VAD duration per surviving note")

    def plot_mean_length_duration_per_surviving_note(ax) -> None:
        duration_bar_colors = np.where(note_flagged_by_duration_mask, "red", "black")
        ax.bar(surviving_note_start_times_mean_length, mean_length_duration_seconds_per_surviving_note, width=mean_length_duration_seconds_per_surviving_note, align="edge", color=duration_bar_colors)
        ax.axhline(mean_length_duration_threshold_per_surviving_note, color="red", linewidth=1.0, linestyle="--")
        ax.set_xlim(0, spectrum_magnitude_after_mean_length.shape[1] * hop_length / sample_rate)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Post-mean-length duration (s)")
        ax.set_title("Post-mean-length duration per surviving note")

    def plot_mean_length_duration_per_surviving_note_combined(ax) -> None:
        combined_duration_bar_colors = np.where(note_flagged_for_deexpansion_mask, "red", "black")
        ax.bar(surviving_note_start_times_mean_length, mean_length_duration_seconds_per_surviving_note, width=mean_length_duration_seconds_per_surviving_note, align="edge", color=combined_duration_bar_colors)
        ax.set_xlim(0, spectrum_magnitude_after_mean_length.shape[1] * hop_length / sample_rate)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Post-mean-length duration (s)")
        ax.set_title("Post-mean-length duration per surviving note (combined criterion)")

    def plot_spectrogram_after_deexpansion(ax) -> None:
        deexpansion_subsampling_step = max(1, spectrum_magnitude_after_deexpansion.shape[1] // 1000)
        deexpansion_max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude_after_deexpansion.shape[0] - 1)) + 1
        spectrum_magnitude_after_deexpansion_db = librosa.amplitude_to_db(spectrum_magnitude_after_deexpansion[:deexpansion_max_frequency_bin_index, ::deexpansion_subsampling_step], ref=spectrum_magnitude_after_mean_length.max())
        ax.imshow(spectrum_magnitude_after_deexpansion_db, origin="lower", aspect="auto", extent=[0, spectrum_magnitude_after_deexpansion.shape[1] * hop_length / sample_rate, 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title("Spectrogram after de-expanding the red notes")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    return audio_after_deexpansion, spectrum_magnitude_after_deexpansion, note_flagged_for_deexpansion_mask, [plot_mean_length_vad_duration_ratio, plot_mean_length_duration_per_surviving_note, plot_mean_length_duration_per_surviving_note_combined, plot_spectrogram_after_deexpansion]


def eliminate_weak_pseudo_notes(spectrum_magnitude_after_deexpansion: np.ndarray, approved_frame_mask: np.ndarray, hop_length: int, sample_rate: int, bin_count_above_threshold_per_frame: np.ndarray, file_name: str, phase_after_reduction: np.ndarray, audio_length_samples: int, short_note_removal_output_path: str) -> tuple:
    """
    Final cleanup pass: remove whole notes that are both weak in peak
    magnitude and weak in average bin count (likely leftover noise
    fragments, "pseudo-notes"), plus any remaining isolated frame whose
    normalized max magnitude is essentially zero.

    For each note (contiguous energy run after de-expansion), robust
    (median - k*MAD) thresholds are computed for peak magnitude and for
    mean bin count, using only notes that had a positive VAD duration as
    the reference population. A note is removed if its peak magnitude is
    below the magnitude threshold AND its mean bin count is below the bin
    count threshold.

    Args:
        spectrum_magnitude_after_deexpansion: STFT magnitude after
            deexpand_erroneously_long_notes.
        approved_frame_mask: Boolean frame mask from detect_call_frames
            (pre-expansion), used as the "VAD" duration reference.
        hop_length: STFT hop length (samples).
        sample_rate: Sample rate of the audio (Hz).
        bin_count_above_threshold_per_frame: Per-frame active bin count, as
            returned by count_bins_per_frame_and_entropy.
        file_name: File name, used only for logging.
        phase_after_reduction: Phase (radians) used for reconstruction.
        audio_length_samples: Target length (samples) for the output audio.
        short_note_removal_output_path: Path to write the reconstructed
            audio to.

    Returns:
        A tuple (audio_after_short_quiet_note_removal,
        spectrum_magnitude_after_short_quiet_note_removal,
        short_quiet_note_mask_after_deexpansion, plots):
            audio_after_short_quiet_note_removal: reconstructed waveform,
                this is the final cleaned audio.
            spectrum_magnitude_after_short_quiet_note_removal: final spectrum.
            short_quiet_note_mask_after_deexpansion: boolean mask over the
                post-deexpansion notes, True for the ones removed here.
            plots: list of 4 diagnostic plotting functions.
    """
    weak_magnitude_mad_multiplier = 4
    weak_bin_count_mad_multiplier = 4
    weak_frame_normalized_magnitude_threshold = 0.001

    note_starts_after_deexpansion, note_ends_after_deexpansion, note_start_times_after_deexpansion, note_end_times_after_deexpansion = compute_time_bounds_of_continuous_energy_frames(spectrum_magnitude_after_deexpansion, hop_length, sample_rate)

    duration_seconds_after_deexpansion_per_note = (note_ends_after_deexpansion - note_starts_after_deexpansion + 1) * hop_length / sample_rate
    vad_duration_seconds_after_deexpansion_per_note = np.array([approved_frame_mask[note_start:note_end + 1].sum() for note_start, note_end in zip(note_starts_after_deexpansion, note_ends_after_deexpansion)]) * hop_length / sample_rate
    max_magnitude_after_deexpansion_per_note = np.array([spectrum_magnitude_after_deexpansion[:, note_start:note_end + 1].max() for note_start, note_end in zip(note_starts_after_deexpansion, note_ends_after_deexpansion)])
    mean_bin_count_after_deexpansion_per_note = np.array([bin_count_above_threshold_per_frame[note_start:note_end + 1].mean() for note_start, note_end in zip(note_starts_after_deexpansion, note_ends_after_deexpansion)])

    note_with_positive_vad_duration_mask_after_deexpansion = vad_duration_seconds_after_deexpansion_per_note > 0
    median_duration_after_deexpansion = np.median(duration_seconds_after_deexpansion_per_note[note_with_positive_vad_duration_mask_after_deexpansion])
    median_max_magnitude_after_deexpansion = np.median(max_magnitude_after_deexpansion_per_note[note_with_positive_vad_duration_mask_after_deexpansion])
    max_magnitude_mad_after_deexpansion = np.median(np.abs(max_magnitude_after_deexpansion_per_note[note_with_positive_vad_duration_mask_after_deexpansion] - median_max_magnitude_after_deexpansion))
    short_note_max_magnitude_threshold = median_max_magnitude_after_deexpansion - weak_magnitude_mad_multiplier * max_magnitude_mad_after_deexpansion
    median_bin_count_after_deexpansion = np.median(mean_bin_count_after_deexpansion_per_note[note_with_positive_vad_duration_mask_after_deexpansion])
    bin_count_mad_after_deexpansion = np.median(np.abs(mean_bin_count_after_deexpansion_per_note[note_with_positive_vad_duration_mask_after_deexpansion] - median_bin_count_after_deexpansion))
    short_note_low_bin_count_threshold = median_bin_count_after_deexpansion - weak_bin_count_mad_multiplier * bin_count_mad_after_deexpansion

    short_quiet_note_mask_after_deexpansion = (max_magnitude_after_deexpansion_per_note < short_note_max_magnitude_threshold) & (mean_bin_count_after_deexpansion_per_note < short_note_low_bin_count_threshold)
    short_quiet_note_start_times_after_deexpansion = note_start_times_after_deexpansion[short_quiet_note_mask_after_deexpansion]
    short_quiet_note_end_times_after_deexpansion = note_end_times_after_deexpansion[short_quiet_note_mask_after_deexpansion]
    short_quiet_note_starts_after_deexpansion = note_starts_after_deexpansion[short_quiet_note_mask_after_deexpansion]
    short_quiet_note_ends_after_deexpansion = note_ends_after_deexpansion[short_quiet_note_mask_after_deexpansion]

    normalized_max_magnitude_per_frame = spectrum_magnitude_after_deexpansion.max(axis=0) / (spectrum_magnitude_after_deexpansion.max() + 1e-12)
    isolated_weak_magnitude_frame_mask = normalized_max_magnitude_per_frame < weak_frame_normalized_magnitude_threshold

    spectrum_magnitude_after_short_quiet_note_removal = spectrum_magnitude_after_deexpansion.copy()
    for short_quiet_note_start, short_quiet_note_end in zip(short_quiet_note_starts_after_deexpansion, short_quiet_note_ends_after_deexpansion):
        spectrum_magnitude_after_short_quiet_note_removal[:, short_quiet_note_start:short_quiet_note_end + 1] = 0
    spectrum_magnitude_after_short_quiet_note_removal[:, isolated_weak_magnitude_frame_mask] = 0

    audio_after_short_quiet_note_removal = librosa.istft(spectrum_magnitude_after_short_quiet_note_removal * np.exp(1j * phase_after_reduction), hop_length=hop_length, window="hann", center=True, length=audio_length_samples).astype(np.float32)
    sf.write(short_note_removal_output_path, audio_after_short_quiet_note_removal, sample_rate)
    print(f"  [{file_name}] audio after short/quiet note removal saved to {short_note_removal_output_path} | {short_quiet_note_mask_after_deexpansion.sum()}/{short_quiet_note_mask_after_deexpansion.size} notes removed | {isolated_weak_magnitude_frame_mask.sum()} frames zeroed for normalized magnitude < {weak_frame_normalized_magnitude_threshold}")

    def plot_max_magnitude_per_note_after_deexpansion(ax) -> None:
        low_magnitude_note_mask_after_deexpansion = max_magnitude_after_deexpansion_per_note < short_note_max_magnitude_threshold
        max_magnitude_bar_colors_after_deexpansion = np.where(low_magnitude_note_mask_after_deexpansion, "red", "black")
        ax.bar(note_start_times_after_deexpansion, max_magnitude_after_deexpansion_per_note, width=duration_seconds_after_deexpansion_per_note, align="edge", color=max_magnitude_bar_colors_after_deexpansion)
        ax.axhline(short_note_max_magnitude_threshold, color="red", linewidth=1.0, linestyle="--")
        ax.set_xlim(0, spectrum_magnitude_after_deexpansion.shape[1] * hop_length / sample_rate)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Max magnitude after de-expansion")
        ax.set_title("Max magnitude per note after de-expansion")

    def plot_bin_count_per_note_after_deexpansion(ax) -> None:
        low_bin_count_note_mask_after_deexpansion = mean_bin_count_after_deexpansion_per_note < short_note_low_bin_count_threshold
        bin_count_bar_colors_after_deexpansion = np.where(low_bin_count_note_mask_after_deexpansion, "red", "black")
        ax.bar(note_start_times_after_deexpansion, mean_bin_count_after_deexpansion_per_note, width=duration_seconds_after_deexpansion_per_note, align="edge", color=bin_count_bar_colors_after_deexpansion)
        ax.axhline(short_note_low_bin_count_threshold, color="red", linewidth=1.0, linestyle="--")
        ax.set_xlim(0, spectrum_magnitude_after_deexpansion.shape[1] * hop_length / sample_rate)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean bin count above threshold")
        ax.set_title("Bin count per note after de-expansion")

    def plot_spectrogram_before_short_quiet_note_removal(ax) -> None:
        pre_removal_subsampling_step = max(1, spectrum_magnitude_after_deexpansion.shape[1] // 1000)
        pre_removal_max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude_after_deexpansion.shape[0] - 1)) + 1
        spectrum_magnitude_after_deexpansion_db = librosa.amplitude_to_db(spectrum_magnitude_after_deexpansion[:pre_removal_max_frequency_bin_index, ::pre_removal_subsampling_step], ref=spectrum_magnitude_after_deexpansion.max())
        ax.imshow(spectrum_magnitude_after_deexpansion_db, origin="lower", aspect="auto", extent=[0, spectrum_magnitude_after_deexpansion.shape[1] * hop_length / sample_rate, 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        for short_quiet_note_start_time, short_quiet_note_end_time in zip(short_quiet_note_start_times_after_deexpansion, short_quiet_note_end_times_after_deexpansion):
            ax.axvline(short_quiet_note_start_time, color="red", linewidth=0.7)
            ax.axvline(short_quiet_note_end_time, color="red", linewidth=0.7)
        ax.set_title("Spectrogram before removing short/quiet notes")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    def plot_final_spectrogram(ax) -> None:
        final_spectrogram_subsampling_step = max(1, spectrum_magnitude_after_short_quiet_note_removal.shape[1] // 1000)
        final_spectrogram_max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (spectrum_magnitude_after_short_quiet_note_removal.shape[0] - 1)) + 1
        spectrum_magnitude_after_short_quiet_note_removal_db = librosa.amplitude_to_db(spectrum_magnitude_after_short_quiet_note_removal[:final_spectrogram_max_frequency_bin_index, ::final_spectrogram_subsampling_step], ref=spectrum_magnitude_after_deexpansion.max())
        ax.imshow(spectrum_magnitude_after_short_quiet_note_removal_db, origin="lower", aspect="auto", extent=[0, spectrum_magnitude_after_short_quiet_note_removal.shape[1] * hop_length / sample_rate, 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title("Final spectrogram")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    return audio_after_short_quiet_note_removal, spectrum_magnitude_after_short_quiet_note_removal, short_quiet_note_mask_after_deexpansion, [plot_max_magnitude_per_note_after_deexpansion, plot_bin_count_per_note_after_deexpansion, plot_spectrogram_before_short_quiet_note_removal, plot_final_spectrogram]


def trim_clean_audio(clean_audio: np.ndarray, clean_audio_spectrum_magnitude: np.ndarray, sample_rate: int, n_fft: int, hop_length: int, file_name: str, trim_output_path: str, audio_length_samples: int) -> tuple:
    """
    Remove silence by dropping every STFT frame with zero magnitude
    entirely (rather than just zeroing it), then reassemble the kept
    frames back-to-back via manual Hann-window overlap-add, so the
    surviving call segments end up butted against each other with no gaps
    between them.

    Args:
        clean_audio: The final cleaned waveform (e.g. output of
            eliminate_weak_pseudo_notes).
        clean_audio_spectrum_magnitude: STFT magnitude of clean_audio,
            used to decide which frames have energy.
        sample_rate: Sample rate of the audio (Hz).
        n_fft: FFT size used for the STFT.
        hop_length: STFT hop length (samples).
        file_name: File name, used only for logging.
        trim_output_path: Path to write the trimmed audio to.
        audio_length_samples: Original (untrimmed) audio length (samples),
            used only for the diagnostic plot's time axis.

    Returns:
        A tuple (trimmed_audio, trimmed_spectrum_magnitude, plots):
            trimmed_audio: waveform with silent frames removed and the
                remaining frames concatenated with no gaps.
            trimmed_spectrum_magnitude: clean_audio_spectrum_magnitude
                restricted to the kept frames.
            plots: list with 1 diagnostic plotting function.
    """
    valid_frame_indices = np.flatnonzero(clean_audio_spectrum_magnitude.max(axis=0) > 0)

    clean_audio_padded_for_stft = np.pad(clean_audio, n_fft // 2)
    hann_window = np.hanning(n_fft).astype(np.float32)

    if valid_frame_indices.size > 0:
        trimmed_audio = np.zeros(valid_frame_indices.size * hop_length + n_fft, dtype=np.float32)
        window_sum = np.zeros_like(trimmed_audio)

        for output_position, valid_frame_index in enumerate(valid_frame_indices):
            block_start_sample = valid_frame_index * hop_length
            original_block = clean_audio_padded_for_stft[block_start_sample:block_start_sample + n_fft]
            output_start_position = output_position * hop_length

            trimmed_audio[output_start_position:output_start_position + n_fft] += original_block * hann_window
            window_sum[output_start_position:output_start_position + n_fft] += hann_window

        trimmed_audio /= np.maximum(window_sum, 1e-8)
        trimmed_audio = trimmed_audio[n_fft // 2:-(n_fft // 2)]
    else:
        trimmed_audio = np.zeros(0, dtype=np.float32)

    sf.write(trim_output_path, trimmed_audio, sample_rate)
    print(f"  [{file_name}] trimming done | {valid_frame_indices.size}/{clean_audio_spectrum_magnitude.shape[1]} frames kept | audio saved to {trim_output_path}")

    trimmed_spectrum_magnitude = clean_audio_spectrum_magnitude[:, valid_frame_indices]

    def plot_trimmed_spectrogram(ax) -> None:
        original_frame_count = clean_audio_spectrum_magnitude.shape[1]
        trimmed_spectrum_magnitude_padded = np.zeros((clean_audio_spectrum_magnitude.shape[0], original_frame_count), dtype=np.float32)
        frames_to_copy_count = min(trimmed_spectrum_magnitude.shape[1], original_frame_count)
        trimmed_spectrum_magnitude_padded[:, :frames_to_copy_count] = trimmed_spectrum_magnitude[:, :frames_to_copy_count]

        subsampling_step = max(1, original_frame_count // 1000)
        max_frequency_bin_index = int(lowpass_frequency / (sample_rate / 2) * (trimmed_spectrum_magnitude_padded.shape[0] - 1)) + 1
        db_reference = clean_audio_spectrum_magnitude.max() + 1e-12
        trimmed_spectrum_magnitude_db = librosa.amplitude_to_db(trimmed_spectrum_magnitude_padded[:max_frequency_bin_index, ::subsampling_step], ref=db_reference)

        ax.imshow(trimmed_spectrum_magnitude_db, origin="lower", aspect="auto", extent=[0, audio_length_samples / sample_rate, 0, lowpass_frequency], cmap="gray_r", vmin=-80.0, vmax=0.0)
        ax.set_title(f"Spectrogram after trimming | duration: {trimmed_audio.size / sample_rate:.2f} s")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    return trimmed_audio, trimmed_spectrum_magnitude, [plot_trimmed_spectrogram]


def process_file(args: tuple) -> None:
    """
    Top-level entry point for one file: runs _process_file_internal and
    catches/logs any exception so that a single failing file does not stop
    the whole (parallel) batch.

    Args:
        args: A (species_name, audio_path) tuple.
    """
    try:
        _process_file_internal(args)
    except Exception:
        print(f"ERROR in {args[1]}:\n{traceback.format_exc()}")


def _process_file_internal(args: tuple) -> None:
    """
    Run the full denoising pipeline on a single audio file, saving a
    numbered checkpoint audio file (in a temporary folder, discarded at the
    end) after each major stage, the final clean and trimmed audio, a
    3-panel "original vs. clean vs. clean+trimmed" spectrogram comparison
    figure, and a large grid figure with every stage's diagnostic plots.
    Intermediate arrays are kept only in memory, not saved to disk.

    Pipeline order: standardize_audio -> remove_human_speech -> bandpass ->
    STFT -> detect_dominant_band -> detect_call_frames -> expand_notes
    (checkpoint 1) -> reduce_noise_noisereduce (checkpoint 2) ->
    median_reduction (checkpoint 3) -> filter_bins_by_correlation_and_probability
    (checkpoint 4) -> count_bins_per_frame_and_entropy (checkpoint 5) ->
    note_duration (checkpoint 6) -> centroid_analysis (checkpoint 7) ->
    mean_continuous_band_length (checkpoint 8) ->
    deexpand_erroneously_long_notes (checkpoint 9) ->
    eliminate_weak_pseudo_notes (checkpoint 10, this is the "clean audio") ->
    trim_clean_audio (final trimmed audio).

    Args:
        args: A (species_name, audio_path) tuple.
    """
    max_rows = 6
    n_fft      = 4096
    hop_length = 256

    species_name, audio_path = args
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    print(f"  [{file_name}] starting")

    standardized_audio, sample_rate = standardize_audio(audio_path)
    audio_after_vad, speech_timestamps, speech_mask = remove_human_speech(standardized_audio, sample_rate, file_name)
    filtered_audio = bandpass(audio_after_vad, sample_rate, file_name)

    audio_length_samples = filtered_audio.shape[0]
    spectrum = librosa.stft(filtered_audio, n_fft=n_fft, hop_length=hop_length, window="hann", center=True).astype(np.complex64, copy=False)
    spectrum_magnitude = np.abs(spectrum).astype(np.float32, copy=False)
    full_fft_freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    removed_speech_label = ", ".join(f"[{speech_segment['start']:.1f}s, {speech_segment['end']:.1f}s]" for speech_segment in speech_timestamps) if speech_timestamps else "none"

    def plot_original_spectrogram(ax) -> None:
        subsampling_step = max(1, spectrum_magnitude.shape[1] // 1000)
        spectrum_magnitude_db = librosa.amplitude_to_db(spectrum_magnitude[:, ::subsampling_step], ref=spectrum_magnitude.max())
        librosa.display.specshow(spectrum_magnitude_db, sr=sample_rate, hop_length=hop_length * subsampling_step, x_axis="s", y_axis="hz", ax=ax, cmap="gray_r", fmax=lowpass_frequency, vmin=-80.0, vmax=0.0)
        ax.set_title(f"Original spectrogram (post VAD+filters) | speech removed: {removed_speech_label}")
        ax.set_ylim(0, lowpass_frequency)

    time_seconds = librosa.frames_to_time(np.arange(spectrum_magnitude.shape[1]), sr=sample_rate, hop_length=hop_length)
    dominant_band_min_frequencies, dominant_band_max_frequencies, dominant_band_plots = detect_dominant_band(spectrum_magnitude, time_seconds, full_fft_freqs)
    spectral_concentration, spectrum_magnitude_after_call_detection, approved_frame_mask, max_magnitude_per_frame, call_detection_plots = detect_call_frames(spectrum_magnitude, sample_rate, time_seconds, dominant_band_min_frequencies, dominant_band_max_frequencies, full_fft_freqs)

    audio_output_folder = tempfile.mkdtemp(prefix=f"{file_name}_")

    audio_after_schmitt, schmitt_spectrum_magnitude, post_schmitt_mask, notes, max_correlation_per_bin_note_expansion, note_expansion_plots = expand_notes(spectrum_magnitude, spectrum, approved_frame_mask, sample_rate, time_seconds, dominant_band_min_frequencies, dominant_band_max_frequencies, hop_length, file_name, os.path.join(audio_output_folder, f"{file_name}_1.wav"), audio_length_samples, full_fft_freqs)

    noise_reduced_audio, spectrum_after_reduction, spectrum_magnitude_after_reduction, noisereduce_plots = reduce_noise_noisereduce(spectrum_magnitude, spectrum, post_schmitt_mask, audio_after_schmitt, sample_rate, n_fft, hop_length, file_name, os.path.join(audio_output_folder, f"{file_name}_2.wav"), audio_length_samples)

    phase_after_reduction = np.angle(spectrum_after_reduction)
    audio_after_median, spectrum_magnitude_after_median, median_reduction_plots = median_reduction(spectrum_magnitude, spectrum_magnitude_after_reduction, phase_after_reduction, post_schmitt_mask, sample_rate, hop_length, file_name, os.path.join(audio_output_folder, f"{file_name}_3.wav"), audio_length_samples)

    spectrum_magnitude_after_correlation_noise_filter, correlation_probability_filter_plots = filter_bins_by_correlation_and_probability(spectrum_magnitude, spectrum_magnitude_after_median, phase_after_reduction, sample_rate, hop_length, file_name, os.path.join(audio_output_folder, f"{file_name}_4.wav"), audio_length_samples, full_fft_freqs, max_correlation_per_bin_note_expansion)

    count_stage_audio_path = os.path.join(audio_output_folder, f"{file_name}_5.wav")
    audio_after_count, spectrum_magnitude_after_count, threshold_approved_frame_mask, bin_count_above_threshold_per_frame, peak_note_frame_count, bin_count_entropy_plots = count_bins_per_frame_and_entropy(spectrum_magnitude_after_correlation_noise_filter, phase_after_reduction, notes, sample_rate, hop_length, file_name, count_stage_audio_path, audio_length_samples, full_fft_freqs)

    duration_stage_audio_path = os.path.join(audio_output_folder, f"{file_name}_6.wav")
    audio_after_duration, spectrum_magnitude_after_duration, pixels_included_by_expansion_mask, note_duration_plots = note_duration(spectrum_magnitude_after_correlation_noise_filter, full_fft_freqs, max_correlation_per_bin_note_expansion, threshold_approved_frame_mask, bin_count_above_threshold_per_frame, peak_note_frame_count, hop_length, sample_rate, file_name, phase_after_reduction, audio_length_samples, duration_stage_audio_path)

    centroid_stage_audio_path = os.path.join(audio_output_folder, f"{file_name}_7.wav")
    audio_after_centroid, spectrum_magnitude_after_centroid, frame_rejected_by_expansion_mask, centroid_analysis_plots = centroid_analysis(spectrum_magnitude_after_correlation_noise_filter, spectrum_magnitude_after_duration, full_fft_freqs, hop_length, sample_rate, pixels_included_by_expansion_mask, file_name, phase_after_reduction, audio_length_samples, centroid_stage_audio_path)

    mean_length_stage_audio_path = os.path.join(audio_output_folder, f"{file_name}_8.wav")
    audio_after_mean_length, spectrum_magnitude_after_mean_length, frame_rejected_by_mean_length_mask, mean_band_length_plots = mean_continuous_band_length(spectrum_magnitude_after_centroid, spectrum_magnitude_after_correlation_noise_filter, pixels_included_by_expansion_mask, threshold_approved_frame_mask, hop_length, sample_rate, file_name, phase_after_reduction, audio_length_samples, mean_length_stage_audio_path)

    deexpansion_stage_audio_path = os.path.join(audio_output_folder, f"{file_name}_9.wav")
    audio_after_deexpansion, spectrum_magnitude_after_deexpansion, note_flagged_for_deexpansion_mask, deexpansion_plots = deexpand_erroneously_long_notes(spectrum_magnitude_after_mean_length, approved_frame_mask, hop_length, sample_rate, full_fft_freqs, dominant_band_min_frequencies, dominant_band_max_frequencies, file_name, phase_after_reduction, audio_length_samples, deexpansion_stage_audio_path)

    short_note_removal_stage_audio_path = os.path.join(audio_output_folder, f"{file_name}_10.wav")
    audio_after_short_quiet_note_removal, spectrum_magnitude_after_short_quiet_note_removal, short_quiet_note_mask_after_deexpansion, pseudo_note_removal_plots = eliminate_weak_pseudo_notes(spectrum_magnitude_after_deexpansion, approved_frame_mask, hop_length, sample_rate, bin_count_above_threshold_per_frame, file_name, phase_after_reduction, audio_length_samples, short_note_removal_stage_audio_path)

    last_saved_audio_path = short_note_removal_stage_audio_path

    # CLEAN AUDIO
    clean_audio_output_folder = os.path.join("clean_audio", species_name)
    os.makedirs(clean_audio_output_folder, exist_ok=True)
    shutil.copy2(last_saved_audio_path, os.path.join(clean_audio_output_folder, os.path.basename(audio_path)))
    shutil.rmtree(audio_output_folder, ignore_errors=True)

    # TRIMMING THE CLEAN AUDIO
    trimmed_audio_output_folder = os.path.join("clean_audio_trimmed", species_name)
    os.makedirs(trimmed_audio_output_folder, exist_ok=True)
    trim_output_path = os.path.join(trimmed_audio_output_folder, f"{file_name}.wav")
    trimmed_audio, trimmed_spectrum_magnitude, trimming_plots = trim_clean_audio(audio_after_short_quiet_note_removal, spectrum_magnitude_after_short_quiet_note_removal, sample_rate, n_fft, hop_length, file_name, trim_output_path, audio_length_samples)

    # CLEAN SPECTROGRAMS
    clean_spectrogram_output_folder = os.path.join("clean_spectrograms", species_name)
    os.makedirs(clean_spectrogram_output_folder, exist_ok=True)

    full_original_spectrum_magnitude = np.abs(librosa.stft(standardized_audio, n_fft=n_fft, hop_length=hop_length, window="hann", center=True)).astype(np.float32, copy=False)
    non_speech_sample_indices = np.flatnonzero(~speech_mask)
    final_frame_sample_positions = np.clip(np.arange(spectrum_magnitude_after_short_quiet_note_removal.shape[1]) * hop_length, 0, audio_length_samples - 1)
    final_frame_original_sample_positions = non_speech_sample_indices[final_frame_sample_positions]
    final_frame_original_frame_indices = np.clip(np.round(final_frame_original_sample_positions / hop_length).astype(int), 0, full_original_spectrum_magnitude.shape[1] - 1)

    clean_spectrum_magnitude_original_axis = np.zeros_like(full_original_spectrum_magnitude)
    clean_spectrum_magnitude_original_axis[:, final_frame_original_frame_indices] = spectrum_magnitude_after_short_quiet_note_removal

    # Padding ONLY for the trimming plot
    trimmed_spectrum_magnitude_original_axis = np.zeros_like(full_original_spectrum_magnitude)
    trimmed_frames_to_copy_count = min(trimmed_spectrum_magnitude.shape[1], trimmed_spectrum_magnitude_original_axis.shape[1])
    trimmed_spectrum_magnitude_original_axis[:, :trimmed_frames_to_copy_count] = trimmed_spectrum_magnitude[:, :trimmed_frames_to_copy_count]

    original_axis_subsampling_step = max(1, full_original_spectrum_magnitude.shape[1] // 1000)
    full_original_spectrum_magnitude_db = librosa.amplitude_to_db(full_original_spectrum_magnitude[:, ::original_axis_subsampling_step], ref=full_original_spectrum_magnitude.max())
    clean_spectrum_magnitude_original_axis_db = librosa.amplitude_to_db(clean_spectrum_magnitude_original_axis[:, ::original_axis_subsampling_step], ref=full_original_spectrum_magnitude.max())
    trimmed_spectrum_magnitude_original_axis_db = librosa.amplitude_to_db(trimmed_spectrum_magnitude_original_axis[:, ::original_axis_subsampling_step], ref=full_original_spectrum_magnitude.max())

    clean_spectrograms_figure, (original_spectrogram_axis, clean_spectrogram_axis, trimmed_spectrogram_axis) = plt.subplots(3, 1, figsize=(12, 15))

    librosa.display.specshow(full_original_spectrum_magnitude_db, sr=sample_rate, hop_length=hop_length * original_axis_subsampling_step, x_axis="s", y_axis="hz", ax=original_spectrogram_axis, cmap="gray_r", vmin=-80.0, vmax=0.0)
    librosa.display.specshow(clean_spectrum_magnitude_original_axis_db, sr=sample_rate, hop_length=hop_length * original_axis_subsampling_step, x_axis="s", y_axis="hz", ax=clean_spectrogram_axis, cmap="gray_r", vmin=-80.0, vmax=0.0)
    librosa.display.specshow(trimmed_spectrum_magnitude_original_axis_db, sr=sample_rate, hop_length=hop_length * original_axis_subsampling_step, x_axis="s", y_axis="hz", ax=trimmed_spectrogram_axis, cmap="gray_r", vmin=-80.0, vmax=0.0)

    frequencies_with_energy_mask = (full_original_spectrum_magnitude > 0).any(axis=1)
    max_frequency_with_energy = full_fft_freqs[frequencies_with_energy_mask].max() if frequencies_with_energy_mask.any() else 0.0
    clean_spectrograms_y_axis_upper_limit = min(lowpass_frequency, max_frequency_with_energy)

    original_spectrogram_axis.set_title("Original spectrogram")
    clean_spectrogram_axis.set_title("Clean spectrogram")
    trimmed_spectrogram_axis.set_title("Clean spectrogram after trimming")

    original_spectrogram_axis.set_xlabel("")
    clean_spectrogram_axis.set_xlabel("")

    original_spectrogram_axis.set_ylim(0, clean_spectrograms_y_axis_upper_limit)
    clean_spectrogram_axis.set_ylim(0, clean_spectrograms_y_axis_upper_limit)
    trimmed_spectrogram_axis.set_ylim(0, clean_spectrograms_y_axis_upper_limit)

    clean_spectrograms_figure.suptitle(file_name, fontsize=14, fontweight="bold")
    clean_spectrograms_figure.tight_layout(rect=[0, 0, 1, 0.97])
    clean_spectrograms_figure.subplots_adjust(hspace=0.35)

    original_axis_position = original_spectrogram_axis.get_position()
    clean_axis_position = clean_spectrogram_axis.get_position()
    trimmed_axis_position = trimmed_spectrogram_axis.get_position()

    clean_spectrograms_figure.add_artist(matplotlib.lines.Line2D([min(original_axis_position.x0, clean_axis_position.x0), max(original_axis_position.x1, clean_axis_position.x1)], [(original_axis_position.y0 + clean_axis_position.y1) / 2] * 2, transform=clean_spectrograms_figure.transFigure, color="black", linewidth=1.5))
    clean_spectrograms_figure.add_artist(matplotlib.lines.Line2D([min(clean_axis_position.x0, trimmed_axis_position.x0), max(clean_axis_position.x1, trimmed_axis_position.x1)], [(clean_axis_position.y0 + trimmed_axis_position.y1) / 2] * 2, transform=clean_spectrograms_figure.transFigure, color="black", linewidth=1.5))

    clean_spectrograms_figure.savefig(os.path.join(clean_spectrogram_output_folder, f"{file_name}.png"), dpi=100, bbox_inches="tight")
    plt.close(clean_spectrograms_figure)

    # Trimming becomes the LAST diagnostic plot
    all_plots = [plot_original_spectrogram] + dominant_band_plots + call_detection_plots + note_expansion_plots + noisereduce_plots + median_reduction_plots + correlation_probability_filter_plots + bin_count_entropy_plots + note_duration_plots + centroid_analysis_plots + mean_band_length_plots + deexpansion_plots + pseudo_note_removal_plots + trimming_plots

    total_plots = len(all_plots)
    column_count = int(np.ceil(total_plots / max_rows))
    row_count = min(total_plots, max_rows)

    diagnostics_figure, diagnostics_axes = plt.subplots(row_count, column_count, figsize=(12 * column_count, 6 * row_count), squeeze=False)
    diagnostics_figure.suptitle(file_name, fontsize=14, fontweight="bold")

    for index, plotting_function in enumerate(all_plots):
        plotting_function(diagnostics_axes[index % max_rows, index // max_rows])

    for empty_index in range(total_plots, row_count * column_count):
        diagnostics_axes[empty_index % max_rows, empty_index // max_rows].set_visible(False)

    diagnostics_output_folder = os.path.join("diagnostic_spectrograms", species_name)
    os.makedirs(diagnostics_output_folder, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(diagnostics_output_folder, f"{file_name}.png"), dpi=60, bbox_inches="tight")
    plt.close()

    del spectrum, spectrum_magnitude

    print(f"Done: {file_name}")


if __name__ == "__main__":
    print("Starting...")

    for root_folder in ("diagnostic_spectrograms", "clean_audio", "clean_audio_trimmed", "clean_spectrograms"):
        if os.path.exists(root_folder):
            shutil.rmtree(root_folder, onexc=lambda f, p, _: (os.chmod(p, stat.S_IWRITE), f(p)))
        os.makedirs(root_folder)

    all_files = [
        (species_name, os.path.join("recordings", species_name, file_name))
        for species_name in sorted(os.listdir("recordings"))
        if os.path.isdir(os.path.join("recordings", species_name))
        for file_name in sorted(os.listdir(os.path.join("recordings", species_name)))
        if os.path.splitext(file_name)[1].lower() in {".mp3", ".wav", ".mp4"}
    ]

    # comment the block above and uncomment this one if you want to run just one species
    '''
    selected_species_name = "Boana_faber"
    all_files = [
        (selected_species_name, os.path.join("recordings", selected_species_name, file_name))
        for file_name in os.listdir(os.path.join("recordings", selected_species_name))
        if os.path.splitext(file_name)[1].lower() in {".mp3", ".wav", ".mp4"}
    ]
    '''

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_file, file) for file in all_files]
        for future in futures:
            future.result()