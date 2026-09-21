"""
CNN species classifier for anuran call spectrograms, trained and evaluated
with leave-one-out cross-validation (LOOCV): the model is retrained once per
file, holding that one file out as the test set, over every file in the
dataset.

Spectrograms are precomputed once per audio folder into log-mel .npz
files (generate_full_spectrograms_from_folder / load_full_spectrograms),
then sliced into fixed-length, non-overlapping time chunks
(generate_chunks_and_masks_from_spectrogram); a boolean mask marks which
frames of the last chunk are real vs. zero-padding, so the model's custom
masked layers (MaskedConvolution, MaskedGlobalAveragePooling) can ignore
padding at both the convolutional and pooling stages. Every file's chunks
are treated as independent training examples sharing that file's label,
and at test time the model's per-chunk predictions for the held-out file
are averaged into one final prediction.

Multiple audio folders (e.g. different denoising methods) and several
random seeds are evaluated back-to-back when this module is run directly;
each (folder, seed) run is written incrementally to a results file so a
run can be killed and resumed later.

This module also acts as a library for cnn_optimization.py, which
calls generate_full_spectrograms_from_folder, load_full_spectrograms,
initialize_worker and train_and_test_one_file directly, and reads/writes
the module-level spectrograms_folder, EPOCHS and PATIENCE. The config dict
schema (see `config` below) and the per-file result dict schema (see
train_and_test_one_file's return value) are this module's public API and
must stay in sync with any other file that imports it.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import shutil
import multiprocessing
import random
import logging
import datetime
import math
import traceback
import numpy as np
import librosa
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

audio_folders = ["clean_audio_trimmed", "clean_audio", "recordings", "audio_noisereduce", "audio_biodenoising", "audio_mmse"]
seeds_to_test = range(15)
fixed_worker_count = 4
species_to_run = None
sample_rate = 22050
n_mel_bands = 64
fft_size = 1024
hop_length = 512
EPOCHS = 400
PATIENCE = 25

spectrograms_folder = "spectrograms_npz"

# Baseline/default hyperparameters used when this module is run directly (see __main__).
# Keys are this module's public config schema: also used, unchanged, by cnn_optimization.py.
config = {
    "conv_layers": 3,
    "filters_per_conv_layer": [32, 64, 128],
    "conv_kernel_size": 3,
    "pooling_size": 2,
    "dense_layers": 0,
    "neurons_per_dense_layer": 256,
    "dropout": 0.3,
    "batch_size": 8,
    "learning_rate": 0.001,
    "l2_regularization": 0.0001,
}


def generate_full_spectrograms_from_folder(audio_folder):
    """
    Compute a log-mel spectrogram for every audio file under audio_folder
    (organized as audio_folder/<species_name>/<file_name>), min-max
    normalize each one to [0, 1] independently, and save it as a
    single-array .npz file under spectrograms_folder/<species_name>/.

    Args:
        audio_folder: Path to a folder with one subfolder per species,
            each containing that species' audio files.
    """
    species_names = sorted(os.listdir(audio_folder))
    labeled_files = []
    for species_name in species_names:
        species_path = os.path.join(audio_folder, species_name)
        for file_name in sorted(os.listdir(species_path)):
            labeled_files.append((os.path.join(species_path, file_name), species_name, file_name))
    for path, label, name in labeled_files:
        waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
        power_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=fft_size, hop_length=hop_length, n_mels=n_mel_bands)
        db_spectrogram = librosa.power_to_db(power_spectrogram, amin=1e-10)
        min_value = db_spectrogram.min()
        max_value = db_spectrogram.max()
        normalized_spectrogram = ((db_spectrogram - min_value) / (max_value - min_value) if max_value > min_value else np.zeros_like(db_spectrogram)).astype(np.float32)
        base_name = os.path.splitext(name)[0]
        species_folder = os.path.join(spectrograms_folder, label)
        os.makedirs(species_folder, exist_ok=True)
        np.savez(os.path.join(species_folder, f"{base_name}.npz"), spectrogram=normalized_spectrogram)
        print(f"spectrogram generated for {name} ({label})")


def load_full_spectrograms():
    """
    Load every precomputed spectrogram under spectrograms_folder back into
    memory (as produced by generate_full_spectrograms_from_folder).

    Returns:
        A tuple (full_spectrograms, text_labels, file_names, species_names):
            full_spectrograms: list of 2D log-mel spectrogram arrays
                (n_mel_bands, n_frames), one per file, in the same order as
                file_names.
            text_labels: species name (string) per file, same order.
            file_names: spectrogram file name (e.g. "foo.npz") per file.
            species_names: sorted list of every species name found (the
                label vocabulary).
    """
    species_names = sorted(os.listdir(spectrograms_folder))
    full_spectrograms = []
    text_labels = []
    file_names = []
    for species_name in species_names:
        species_path = os.path.join(spectrograms_folder, species_name)
        for spectrogram_file_name in sorted(os.listdir(species_path)):
            full_spectrogram = np.load(os.path.join(species_path, spectrogram_file_name))["spectrogram"]
            full_spectrograms.append(full_spectrogram)
            text_labels.append(species_name)
            file_names.append(spectrogram_file_name)
    return full_spectrograms, text_labels, file_names, species_names


def generate_chunks_and_masks_from_spectrogram(full_spectrogram, window_size_in_frames):
    """
    Split one spectrogram into fixed-length, non-overlapping time chunks,
    zero-padding the last chunk if needed, and build a matching boolean
    mask (1 = real frame, 0 = padding) for each chunk.

    Args:
        full_spectrogram: 2D array (n_mel_bands, n_frames).
        window_size_in_frames: Number of time frames per chunk.

    Returns:
        A list of (chunk, mask) tuples: chunk has shape
        (n_mel_bands, window_size_in_frames); mask has shape
        (window_size_in_frames,).
    """
    real_frame_count = full_spectrogram.shape[1]
    chunk_count = math.ceil(real_frame_count / window_size_in_frames)
    chunks_and_masks = []
    for chunk_index in range(chunk_count):
        chunk_start = chunk_index * window_size_in_frames
        chunk_end = min(chunk_start + window_size_in_frames, real_frame_count)
        real_frames_in_chunk = chunk_end - chunk_start
        padding_frames_in_chunk = window_size_in_frames - real_frames_in_chunk
        padded_chunk = np.pad(full_spectrogram[:, chunk_start:chunk_end], ((0, 0), (0, padding_frames_in_chunk)), mode="constant", constant_values=0.0)
        chunk_mask = np.concatenate([np.ones(real_frames_in_chunk, dtype=np.float32), np.zeros(padding_frames_in_chunk, dtype=np.float32)])
        chunks_and_masks.append((padded_chunk, chunk_mask))
    return chunks_and_masks


class EpochMonitor(tf.keras.callbacks.Callback):
    """
    Keras callback that tracks the best (lowest) training loss seen so far
    for one LOOCV run and logs progress every 50 epochs.
    """

    def __init__(self, test_file_name):
        """
        Args:
            test_file_name: Name of the file held out for testing in this
                run, used only for logging.
        """
        super().__init__()
        self.test_file_name = test_file_name

    def on_train_begin(self, logs=None):
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.last_epoch_time = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs["loss"]
        current_time = datetime.datetime.now()
        epoch_seconds = (current_time - self.last_epoch_time).total_seconds()
        self.last_epoch_time = current_time
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch + 1
        if (epoch + 1) % 50 == 0:
            print(f"[PID {os.getpid()}] {self.test_file_name} - epoch {epoch + 1}/{EPOCHS} - loss: {current_loss:.3f} - best loss: {self.best_loss:.3f} - epoch_duration: {epoch_seconds:.1f}s")


class MaskedGlobalAveragePooling(tf.keras.layers.Layer):
    """
    Global average pooling over height and time, but averaging only over
    the time frames marked valid by a temporal mask (so zero-padded chunk
    frames don't dilute the average).
    """

    def call(self, inputs):
        """
        Args:
            inputs: A (activation_map, temporal_mask) pair. activation_map
                has shape (batch, height, time, channels); temporal_mask
                has shape (batch, time), 1 for real frames and 0 for padding.

        Returns:
            Pooled activations, shape (batch, channels).
        """
        activation_map, temporal_mask = inputs
        expanded_mask = temporal_mask[:, tf.newaxis, :, tf.newaxis]
        masked_sum = tf.reduce_sum(activation_map * expanded_mask, axis=[1, 2])
        height = tf.cast(tf.shape(activation_map)[1], tf.float32)
        valid_frame_count = tf.reduce_sum(temporal_mask, axis=1, keepdims=True)
        return masked_sum / (height * valid_frame_count)


class MaskedConvolution(tf.keras.layers.Layer):
    """
    2D convolution that is aware of a spatial mask: padded (masked-out)
    positions do not contribute to the convolution, and the output at each
    position is re-normalized by how much of the kernel's receptive field
    was actually valid there, so results near the mask's edge aren't
    biased low. The mask itself is propagated forward (via a fixed,
    all-ones, non-trainable convolution) so the next layer knows which
    positions are still valid.
    """

    def __init__(self, filter_count, kernel_size, l2_regularization, **kwargs):
        """
        Args:
            filter_count: Number of output channels (Conv2D filters).
            kernel_size: Convolution kernel size (assumed square).
            l2_regularization: L2 weight regularization strength for the
                content convolution's kernel.
        """
        super().__init__(**kwargs)
        self.filter_count = filter_count
        self.kernel_size = kernel_size
        self.l2_regularization = l2_regularization
        self.kernel_area = float(kernel_size * kernel_size)

    def build(self, input_shape):
        self.content_convolution = tf.keras.layers.Conv2D(self.filter_count, self.kernel_size, padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization))
        self.mask_convolution = tf.keras.layers.Conv2D(1, self.kernel_size, padding="same", use_bias=False, kernel_initializer="ones", trainable=False)
        self.bias = self.add_weight(name="bias", shape=(self.filter_count,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        """
        Args:
            inputs: An (activation_map, mask_2d) pair. activation_map has
                shape (batch, height, width, channels); mask_2d has shape
                (batch, height, width, 1), 1 for valid positions and 0 for
                padding.

        Returns:
            A (final_output, valid_mask) pair: final_output is the
            normalized, biased, ReLU-activated convolution result;
            valid_mask is 1 wherever the kernel's receptive field covered
            at least one valid input position, 0 elsewhere.
        """
        activation_map, mask_2d = inputs
        convolution_output = self.content_convolution(activation_map * mask_2d)
        mask_sum = self.mask_convolution(mask_2d)
        normalization_factor = self.kernel_area / (mask_sum + 1e-8)
        valid_mask = tf.cast(mask_sum > 0, tf.float32)
        final_output = tf.nn.relu(convolution_output * normalization_factor * valid_mask + self.bias)
        return final_output, valid_mask


def build_classifier(config, spectrogram_height, spectrogram_width, class_count):
    """
    Build and compile the masked CNN classifier: a stack of
    MaskedConvolution + max-pooling blocks (mask max-pooled alongside the
    activations), followed by MaskedGlobalAveragePooling, an optional
    stack of dense layers with dropout, and a softmax output layer.

    Args:
        config: A config dict (see module docstring for the schema).
        spectrogram_height: Height (n_mel_bands) of the input spectrograms.
        spectrogram_width: Width (time frames) of the input spectrograms.
        class_count: Number of output classes (species).

    Returns:
        A compiled `tf.keras.Model` taking [spectrogram, mask] as input.
    """
    spectrogram_input = tf.keras.layers.Input(shape=(spectrogram_height, spectrogram_width, 1))
    mask_input = tf.keras.layers.Input(shape=(spectrogram_width,))
    current_mask = tf.keras.layers.Lambda(lambda mask: tf.tile(tf.reshape(mask, [-1, 1, spectrogram_width, 1]), [1, spectrogram_height, 1, 1]))(mask_input)
    activation_map = spectrogram_input
    for layer_index in range(config["conv_layers"]):
        activation_map, current_mask = MaskedConvolution(config["filters_per_conv_layer"][layer_index], config["conv_kernel_size"], config["l2_regularization"])([activation_map, current_mask])
        activation_map = tf.keras.layers.MaxPooling2D(config["pooling_size"])(activation_map)
        current_mask = tf.keras.layers.MaxPooling2D(config["pooling_size"])(current_mask)
    flattened_reduced_mask = tf.keras.layers.Lambda(lambda mask: mask[:, 0, :, 0])(current_mask)
    current_layer = MaskedGlobalAveragePooling()([activation_map, flattened_reduced_mask])
    for layer_index in range(config["dense_layers"]):
        current_layer = tf.keras.layers.Dense(config["neurons_per_dense_layer"], activation="relu", kernel_regularizer=tf.keras.regularizers.l2(config["l2_regularization"]))(current_layer)
        current_layer = tf.keras.layers.Dropout(config["dropout"])(current_layer)
    output = tf.keras.layers.Dense(class_count, activation="softmax")(current_layer)
    model = tf.keras.Model(inputs=[spectrogram_input, mask_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss="categorical_crossentropy")
    return model


def initialize_worker(config_for_workers, full_spectrograms_for_workers, label_indices_for_workers, file_names_for_workers, species_names_for_workers, random_seed_for_workers):
    """
    Multiprocessing pool initializer: stores everything a worker process
    needs as module-level globals (so train_and_test_one_file, called
    per-task, doesn't need to re-receive the whole dataset every time), and
    seeds every RNG for reproducibility.

    Args:
        config_for_workers: The config dict to train with in this pool.
        full_spectrograms_for_workers: Precomputed spectrograms for every file
            (see load_full_spectrograms).
        label_indices_for_workers: Integer species-label index per file.
        file_names_for_workers: List of all file names in the dataset.
        species_names_for_workers: List of all species names (label vocabulary).
        random_seed_for_workers: Random seed to seed every RNG with.
    """
    for gpu_device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu_device, True)
    global worker_config, worker_full_spectrograms, worker_label_indices, worker_file_names, worker_species_names, worker_random_seed
    worker_config = config_for_workers
    worker_full_spectrograms = full_spectrograms_for_workers
    worker_label_indices = label_indices_for_workers
    worker_file_names = file_names_for_workers
    worker_species_names = species_names_for_workers
    worker_random_seed = random_seed_for_workers
    random.seed(worker_random_seed)
    np.random.seed(worker_random_seed)
    tf.random.set_seed(worker_random_seed)
    print(f"[PID {os.getpid()}] worker initialized")


def train_and_test_one_file(test_file_name):
    """
    Top-level entry point for one LOOCV fold: runs
    _train_and_test_one_file_internal and catches/logs any exception so
    that a single failing file does not stop the whole (parallel) batch.

    Args:
        test_file_name: Name of the file to hold out and test on.

    Returns:
        The result dict from _train_and_test_one_file_internal, or None if
        it raised an exception.
    """
    try:
        return _train_and_test_one_file_internal(test_file_name)
    except Exception:
        print(f"ERROR in {test_file_name}:\n{traceback.format_exc()}")
        return None


def _train_and_test_one_file_internal(test_file_name):
    """
    Train a fresh model on every file except test_file_name, then predict
    on test_file_name's chunks and average the per-chunk probabilities into
    one final prediction. Must be called in a worker process previously
    set up by initialize_worker (it reads that function's module-level
    globals).

    The chunk window size is the minimum spectrogram length among the
    training files, so no training chunk needs padding; test_file_name's
    spectrogram is then chunked with that same window size (its own chunks
    may need padding if it's longer or shorter than that window).

    Args:
        test_file_name: Name of the file to hold out and test on.

    Returns:
        A result dict with keys "file_name", "correct", "correct_species",
        "predicted_species", "predicted_confidence",
        "secondary_predicted_species", "secondary_predicted_confidence".
        This dict's keys are this module's public per-file result schema,
        shared with cnn_optimization.py.
    """
    tf.keras.backend.clear_session()
    class_count = len(worker_species_names)
    test_file_index = worker_file_names.index(test_file_name)
    train_indices = [index for index, name in enumerate(worker_file_names) if name != test_file_name]
    window_size_in_frames = min(worker_full_spectrograms[index].shape[1] for index in train_indices)
    train_spectrograms = []
    train_masks = []
    train_labels = []
    for index in train_indices:
        for chunk, mask in generate_chunks_and_masks_from_spectrogram(worker_full_spectrograms[index], window_size_in_frames):
            train_spectrograms.append(chunk)
            train_masks.append(mask)
            train_labels.append(worker_label_indices[index])
    test_spectrograms = []
    test_masks = []
    for chunk, mask in generate_chunks_and_masks_from_spectrogram(worker_full_spectrograms[test_file_index], window_size_in_frames):
        test_spectrograms.append(chunk)
        test_masks.append(mask)
    train_spectrograms = np.array(train_spectrograms)[..., np.newaxis]
    train_masks = np.array(train_masks)
    train_labels = tf.keras.utils.to_categorical(train_labels, class_count)
    test_spectrograms = np.array(test_spectrograms)[..., np.newaxis]
    test_masks = np.array(test_masks)
    tf.random.set_seed(worker_random_seed)
    model = build_classifier(worker_config, train_spectrograms.shape[1], train_spectrograms.shape[2], class_count)
    epoch_monitor = EpochMonitor(test_file_name)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=PATIENCE, restore_best_weights=True)
    model.fit([train_spectrograms, train_masks], train_labels, batch_size=worker_config["batch_size"], epochs=EPOCHS, verbose=0, callbacks=[epoch_monitor, early_stopping])
    predicted_probabilities = model.predict([test_spectrograms, test_masks], verbose=0).mean(axis=0)
    probability_sorted_indices = np.argsort(predicted_probabilities)[::-1]
    predicted_class = probability_sorted_indices[0]
    secondary_predicted_class = probability_sorted_indices[1]
    correct_class = worker_label_indices[test_file_index]
    correct = bool(predicted_class == correct_class)
    print(f"File {test_file_name} - best epoch: {epoch_monitor.best_epoch} - result: {'CORRECT' if correct else 'WRONG'}")
    return {"file_name": test_file_name, "correct": correct, "correct_species": worker_species_names[correct_class], "predicted_species": worker_species_names[predicted_class], "predicted_confidence": float(predicted_probabilities[predicted_class]), "secondary_predicted_species": worker_species_names[secondary_predicted_class], "secondary_predicted_confidence": float(predicted_probabilities[secondary_predicted_class])}


def generate_seed_result_text(random_seed, per_file_results):
    """
    Format one random seed's full LOOCV results as plain text, in the
    format this module's own __main__ resume logic parses back.

    Args:
        random_seed: The random seed this LOOCV run used.
        per_file_results: List of per-file result dicts (see
            _train_and_test_one_file_internal).

    Returns:
        The formatted text block (a "SEED: ..." header, summary stats, and
        one line per file), ready to be inserted into the results file.
    """
    total_correct = sum(1 for result in per_file_results if result["correct"])
    accuracy_percentage = 100 * total_correct / len(per_file_results)
    result_lines = [f"SEED: {random_seed}\n", f"correct: {total_correct}\n", f"total_files: {len(per_file_results)}\n", f"accuracy_percentage: {accuracy_percentage:.2f}\n", "results_per_file\n"]
    for result in sorted(per_file_results, key=lambda result: result["file_name"]):
        result_lines.append(f"file: {result['file_name']} | correct_species: {result['correct_species']} | answer: {result['predicted_species']} | answer_confidence: {result['predicted_confidence']:.4f} | secondary_answer: {result['secondary_predicted_species']} | secondary_answer_confidence: {result['secondary_predicted_confidence']:.4f} | correct: {result['correct']}\n")
    result_lines.append("\n")
    return "".join(result_lines)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    results_file_path = "cnn.txt"
    run_is_being_resumed = os.path.exists(results_file_path) and os.path.getsize(results_file_path) > 0

    completed_seeds_per_folder = {}
    if run_is_being_resumed:
        folder_being_read = None
        with open(results_file_path, "r") as results_file:
            for line in results_file:
                if line.startswith("folder: "):
                    folder_being_read = line.split("folder: ")[1].strip()
                    completed_seeds_per_folder.setdefault(folder_being_read, set())
                elif line.startswith("SEED: "):
                    completed_seeds_per_folder[folder_being_read].add(int(line.split("SEED: ")[1].strip()))
        print(f"resuming from {results_file_path}")
    else:
        with open(results_file_path, "w") as results_file:
            results_file.write("FIXED HYPERPARAMETERS\n")
            results_file.write("embedding_model: CNN_from_scratch\n")
            results_file.write(f"sample_rate: {sample_rate}\n")
            results_file.write(f"n_mel_bands: {n_mel_bands}\n")
            results_file.write(f"fft_size: {fft_size}\n")
            results_file.write(f"hop_length: {hop_length}\n")
            results_file.write("normalization: log_mel_min_max_per_file\n")
            results_file.write("optimizer: Adam\n")
            results_file.write("loss_function: categorical_crossentropy\n")
            results_file.write("hidden_layer_activation: relu\n")
            results_file.write("output_activation: softmax\n")
            results_file.write(f"early_stopping_patience: {PATIENCE}\n")
            results_file.write(f"max_epoch_cap: {EPOCHS}\n")
            results_file.write(f"problem_class_count: {len(os.listdir(audio_folders[0]))}\n")
            results_file.write(f"run_start_time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            results_file.write("TESTED HYPERPARAMETERS\n")
            for key, value in config.items():
                results_file.write(f"{key}: {value}\n")
            results_file.write("\n")

    for audio_folder in audio_folders:
        already_completed_seeds_this_folder = completed_seeds_per_folder.get(audio_folder, set())
        missing_seeds_this_folder = [seed for seed in seeds_to_test if seed not in already_completed_seeds_this_folder]

        if not missing_seeds_this_folder:
            if os.path.exists(spectrograms_folder):
                shutil.rmtree(spectrograms_folder)
            print(f"folder {audio_folder} already complete, skipping")
            continue

        print(f"folder: {audio_folder}")
        if not already_completed_seeds_this_folder:
            with open(results_file_path, "a") as results_file:
                results_file.write(f"folder: {audio_folder}\n")

        if not os.path.exists(spectrograms_folder):
            generate_full_spectrograms_from_folder(audio_folder)

        full_spectrograms, text_labels, file_names, species_names = load_full_spectrograms()
        print(f"spectrograms loaded from {spectrograms_folder}")

        label_indices = [species_names.index(label) for label in text_labels]
        files_to_test = [file_name for file_name, label in zip(file_names, text_labels) if not species_to_run or label == species_to_run]

        for random_seed in missing_seeds_this_folder:
            random.seed(random_seed)
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

            print(f"starting training pool with {fixed_worker_count} processes for {len(files_to_test)} files - seed {random_seed}")
            with multiprocessing.Pool(fixed_worker_count, initializer=initialize_worker, initargs=(config, full_spectrograms, label_indices, file_names, species_names, random_seed), maxtasksperchild=1) as pool:
                per_file_results = []
                processed_file_count = 0
                for result in pool.imap_unordered(train_and_test_one_file, sorted(files_to_test)):
                    processed_file_count += 1
                    if result is not None:
                        per_file_results.append(result)
                    print(f"[OVERALL PROGRESS] seed {random_seed} - {processed_file_count}/{len(files_to_test)} files done")

            seed_result_text = generate_seed_result_text(random_seed, per_file_results)
            current_file_content = open(results_file_path, "r").read()
            folder_header_index = current_file_content.index(f"folder: {audio_folder}\n")
            next_folder_header = next((f"\nfolder: {other_folder}\n" for other_folder in audio_folders if other_folder != audio_folder and f"\nfolder: {other_folder}\n" in current_file_content[folder_header_index:]), None)
            insertion_index = current_file_content.index(next_folder_header, folder_header_index) if next_folder_header else len(current_file_content)
            updated_content = current_file_content[:insertion_index] + seed_result_text + current_file_content[insertion_index:]
            open(results_file_path, "w").write(updated_content)

            total_correct = sum(1 for result in per_file_results if result["correct"])
            accuracy_percentage = 100 * total_correct / len(per_file_results)
            print(f"LOOCV done - seed {random_seed} - correct: {total_correct}/{len(per_file_results)} - accuracy: {accuracy_percentage:.2f}%")
            print(f"Seed {random_seed} results saved to {results_file_path}")

        shutil.rmtree(spectrograms_folder)
        print(f"{spectrograms_folder} deleted at the end of folder {audio_folder}")