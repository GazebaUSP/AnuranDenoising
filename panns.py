"""
Species classifier built on top of PANNs CNN14 embeddings (a pretrained
audio-tagging model, used purely as a fixed feature extractor: it is never
fine-tuned), trained and evaluated with leave-one-out cross-validation
(LOOCV): a small dense classifier head is retrained once per file, holding
that one file out as the test set, over every file in the dataset.

Unlike cnn.py's from-scratch CNN, there is no spectrogram pipeline here:
every audio file is loaded once, padded/looped to at least 1 second, and
fed whole into PANNs' AudioTagging.inference() to get one fixed-size
embedding vector per file; the classifier head is just a stack of dense +
dropout layers on top of that embedding.

Multiple audio folders (e.g. different denoising methods) and several
random seeds are evaluated back-to-back when this module is run directly;
each (folder, seed) run is written incrementally to a results file so a
run can be killed and resumed later.

This module also acts as a library for panns_optimization.py, which calls
extract_embeddings_from_folder, initialize_worker and
train_and_test_one_file directly, and reads the module-level
embedding_dimension. The config dict schema (see `config` below) and the
per-file result dict schema (see train_and_test_one_file's return value)
are this module's public API and must stay in sync with any other file
that imports it (in particular, they use the same key names as cnn.py's
schema, for consistency across the project).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import multiprocessing
import random
import logging
import datetime
import numpy as np
import librosa
import tensorflow as tf
from panns_inference import AudioTagging
tf.get_logger().setLevel(logging.ERROR)

audio_folders = ["clean_audio_trimmed", "clean_audio", "recordings", "audio_noisereduce", "audio_biodenoising"]
seeds_to_test = range(15)
fixed_worker_count = 1
species_to_run = None
panns_sample_rate = 32000
embedding_dimension = 2048

# Baseline/default hyperparameters used when this module is run directly (see __main__).
# Keys are this module's public config schema: also used, unchanged, by panns_optimization.py,
# and match cnn.py's own config schema for the shared keys (dense_layers, dropout, etc.).
config = {
    "dense_layers": 1,
    "neurons_per_dense_layer": 128,
    "dropout": 0.3,
    "batch_size": 8,
    "learning_rate": 0.002,
    "l2_regularization": 0.001,
}


def extract_embeddings_from_folder(audio_folder, audio_model):
    """
    Extract one fixed-size PANNs embedding per audio file under audio_folder
    (organized as audio_folder/<species_name>/<file_name>).

    Each waveform is loaded at panns_sample_rate and, if shorter than 1
    second, wrap-padded (looped) up to 1 second, since PANNs expects a
    minimum input length; it is then run through audio_model.inference in
    full (no chunking), producing one embedding vector per file.

    Args:
        audio_folder: Path to a folder with one subfolder per species,
            each containing that species' audio files.
        audio_model: A `panns_inference.AudioTagging` instance to run
            inference with.

    Returns:
        A tuple (embeddings, text_labels, file_names, species_names):
            embeddings: array of shape (n_files, embedding_dimension).
            text_labels: species name (string) per file, same order.
            file_names: file name per file, same order.
            species_names: sorted list of every species name found (the
                label vocabulary).
    """
    species_names = sorted(os.listdir(audio_folder))
    labeled_files = []
    for species_name in species_names:
        species_path = os.path.join(audio_folder, species_name)
        for file_name in sorted(os.listdir(species_path)):
            labeled_files.append((os.path.join(species_path, file_name), species_name, file_name))
    embeddings = []
    for path, label, name in labeled_files:
        waveform, _ = librosa.load(path, sr=panns_sample_rate, mono=True)
        min_duration_in_samples = panns_sample_rate * 1
        waveform_with_min_duration = np.pad(waveform, (0, max(0, min_duration_in_samples - len(waveform))), mode="wrap") if len(waveform) > 0 else np.zeros(min_duration_in_samples)
        _, embedding = audio_model.inference(waveform_with_min_duration[np.newaxis, :])
        embeddings.append(embedding[0])
        print(f"embedding extracted for {name} ({label})")
    embeddings = np.array(embeddings)
    text_labels = [label for path, label, name in labeled_files]
    file_names = [name for path, label, name in labeled_files]
    return embeddings, text_labels, file_names, species_names


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
            print(f"[PID {os.getpid()}] {self.test_file_name} - epoch {epoch + 1}/200 - loss: {current_loss:.3f} - best loss: {self.best_loss:.3f} - epoch_duration: {epoch_seconds:.1f}s")


def build_classifier(config, embedding_dimension, class_count):
    """
    Build and compile the dense classifier head on top of a fixed PANNs
    embedding: an optional stack of dense + dropout layers, followed by a
    softmax output layer.

    Args:
        config: A config dict (see module docstring for the schema).
        embedding_dimension: Size of the input embedding vector.
        class_count: Number of output classes (species).

    Returns:
        A compiled `tf.keras.Model` taking the embedding vector as input.
    """
    embedding_input = tf.keras.layers.Input(shape=(embedding_dimension,))
    current_layer = embedding_input
    for layer_index in range(config["dense_layers"]):
        current_layer = tf.keras.layers.Dense(config["neurons_per_dense_layer"], activation="relu", kernel_regularizer=tf.keras.regularizers.l2(config["l2_regularization"]))(current_layer)
        current_layer = tf.keras.layers.Dropout(config["dropout"])(current_layer)
    output = tf.keras.layers.Dense(class_count, activation="softmax")(current_layer)
    model = tf.keras.Model(inputs=embedding_input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss="categorical_crossentropy")
    return model


def initialize_worker(config_for_workers, embeddings_for_workers, label_indices_for_workers, file_names_for_workers, species_names_for_workers, random_seed_for_workers):
    """
    Multiprocessing pool initializer: stores everything a worker process
    needs as module-level globals (so train_and_test_one_file, called
    per-task, doesn't need to re-receive the whole dataset every time), and
    seeds every RNG for reproducibility.

    Args:
        config_for_workers: The config dict to train with in this pool.
        embeddings_for_workers: Precomputed embeddings for every file (see
            extract_embeddings_from_folder).
        label_indices_for_workers: Integer species-label index per file.
        file_names_for_workers: List of all file names in the dataset.
        species_names_for_workers: List of all species names (label
            vocabulary).
        random_seed_for_workers: Random seed to seed every RNG with.
    """
    global worker_config, worker_embeddings, worker_label_indices, worker_file_names, worker_species_names, worker_random_seed
    worker_config = config_for_workers
    worker_embeddings = embeddings_for_workers
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
    Train a fresh classifier head on every file's embedding except
    test_file_name's, then predict on test_file_name's embedding. Must be
    called in a worker process previously set up by initialize_worker (it
    reads that function's module-level globals).

    Args:
        test_file_name: Name of the file to hold out and test on.

    Returns:
        A result dict with keys "file_name", "correct", "correct_species",
        "predicted_species", "predicted_confidence",
        "secondary_predicted_species", "secondary_predicted_confidence".
        This dict's keys are this module's public per-file result schema,
        shared with panns_optimization.py (and matching cnn.py's schema).
    """
    tf.keras.backend.clear_session()
    class_count = len(worker_species_names)
    categorical_labels = tf.keras.utils.to_categorical(worker_label_indices, class_count)
    test_index = worker_file_names.index(test_file_name)
    train_indices = [index for index in range(len(worker_file_names)) if index != test_index]
    train_embeddings = worker_embeddings[train_indices]
    train_labels = categorical_labels[train_indices]
    test_embedding = worker_embeddings[test_index][np.newaxis, ...]
    tf.random.set_seed(worker_random_seed)
    model = build_classifier(worker_config, worker_embeddings.shape[1], class_count)
    epoch_monitor = EpochMonitor(test_file_name)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    model.fit(train_embeddings, train_labels, batch_size=worker_config["batch_size"], epochs=200, verbose=0, callbacks=[epoch_monitor, early_stopping])
    predicted_probabilities = model.predict(test_embedding, verbose=0)[0]
    probability_sorted_indices = np.argsort(predicted_probabilities)[::-1]
    predicted_class = probability_sorted_indices[0]
    secondary_predicted_class = probability_sorted_indices[1]
    correct_class = worker_label_indices[test_index]
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
            train_and_test_one_file).

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
    embeddings_cache_path = "embeddings_cache.npz"
    results_file_path = "panns.txt"
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
            results_file.write("embedding_model: PANNs_CNN14\n")
            results_file.write(f"embedding_dimension: {embedding_dimension}\n")
            results_file.write(f"panns_sample_rate: {panns_sample_rate}\n")
            results_file.write("optimizer: Adam\n")
            results_file.write("loss_function: categorical_crossentropy\n")
            results_file.write("hidden_layer_activation: relu\n")
            results_file.write("output_activation: softmax\n")
            results_file.write("early_stopping_patience: 10\n")
            results_file.write("max_epoch_cap: 200\n")
            results_file.write(f"problem_class_count: {len(os.listdir(audio_folders[0]))}\n")
            results_file.write(f"run_start_time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            results_file.write("TESTED HYPERPARAMETERS\n")
            for key, value in config.items():
                results_file.write(f"{key}: {value}\n")
            results_file.write("\n")

    audio_model = AudioTagging(checkpoint_path=None, device="cpu")

    for audio_folder in audio_folders:
        already_completed_seeds_this_folder = completed_seeds_per_folder.get(audio_folder, set())
        missing_seeds_this_folder = [seed for seed in seeds_to_test if seed not in already_completed_seeds_this_folder]

        if not missing_seeds_this_folder:
            if os.path.exists(embeddings_cache_path):
                os.remove(embeddings_cache_path)
            print(f"folder {audio_folder} already complete, skipping")
            continue

        print(f"folder: {audio_folder}")
        if not already_completed_seeds_this_folder:
            with open(results_file_path, "a") as results_file:
                results_file.write(f"folder: {audio_folder}\n")

        if not os.path.exists(embeddings_cache_path):
            embeddings, text_labels, file_names, species_names = extract_embeddings_from_folder(audio_folder, audio_model)
            np.savez(embeddings_cache_path, embeddings=embeddings, text_labels=text_labels, file_names=file_names, species_names=species_names)

        loaded_cache = np.load(embeddings_cache_path, allow_pickle=True)
        embeddings = loaded_cache["embeddings"]
        text_labels = list(loaded_cache["text_labels"])
        file_names = list(loaded_cache["file_names"])
        species_names = list(loaded_cache["species_names"])
        print(f"embeddings loaded from {embeddings_cache_path}")

        label_indices = [species_names.index(label) for label in text_labels]
        files_to_test = [file_name for file_name, label in zip(file_names, text_labels) if not species_to_run or label == species_to_run]

        for random_seed in missing_seeds_this_folder:
            random.seed(random_seed)
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

            print(f"starting training pool with {fixed_worker_count} processes for {len(files_to_test)} files - seed {random_seed}")
            with multiprocessing.Pool(fixed_worker_count, initializer=initialize_worker, initargs=(config, embeddings, label_indices, file_names, species_names, random_seed), maxtasksperchild=1) as pool:
                per_file_results = []
                sorted_files = sorted(files_to_test)
                for file_index, file_name in enumerate(sorted_files, start=1):
                    result = pool.apply(train_and_test_one_file, (file_name,))
                    per_file_results.append(result)
                    print(f"[OVERALL PROGRESS] seed {random_seed} - {file_index}/{len(sorted_files)} files done")

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

        os.remove(embeddings_cache_path)
        print(f"{embeddings_cache_path} deleted at the end of folder {audio_folder}")