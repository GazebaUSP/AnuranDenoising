"""
Hyperparameter optimizer for a PANNs-embedding-based species classifier,
using Optuna (TPE sampler) to search over a fixed grid of classifier-head
hyperparameters, scored by leave-one-out cross-validation (LOOCV) over the
full set of cleaned/trimmed recordings.

For each candidate config, run_trial_loocv trains and tests one file at a
time (each in its own subprocess, via the `panns` module) and stops the
trial early ("pruned") as soon as it can no longer beat the best config
found so far, to save compute. Progress is written incrementally to a
plain-text results file (results_file_path) so a run can be killed and
resumed later: on restart, __main__ parses that file back into completed
trials and re-registers them with a fresh Optuna study before continuing.

Unlike cnn_optimization.py, there is no architecture grid to search here
(PANNs' embedding extractor is fixed/pretrained and never searched over) —
only the dense classifier head's hyperparameters are tuned. This file's
config dict and per-file result dict use the exact same key names as
`panns.py`'s public API (see that module's docstring, and cnn.py's schema,
which they also match): all three files must be kept in sync if that
schema ever changes.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import ast
import datetime
import multiprocessing
import time
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import panns

audio_folder = "clean_audio_trimmed"
random_seed = 0
total_trials = 30
results_file_path = "panns_config_optimizer.txt"

# Keys of this dict are `panns`'s config API (see module docstring): must match panns.py exactly.
baseline_config = {
    "dense_layers": 1,
    "neurons_per_dense_layer": 128,
    "dropout": 0.5,
    "batch_size": 8,
    "learning_rate": 0.001,
    "l2_regularization": 0.01,
}


def get_hashable_config_key(config):
    """
    Turn a config dict into a hashable, order-independent key, so two
    configs with the same content (regardless of dict insertion order) map
    to the same key. Unlike cnn_optimization.py's version, no value here is
    ever a list, so no extra list-to-tuple conversion is needed.

    Args:
        config: A `panns`-style config dict (see baseline_config for the
            expected keys).

    Returns:
        A tuple of (key, value) pairs sorted by key, safe to use as a dict
        key or set element.
    """
    return tuple(sorted(config.items()))


possible_dense_layer_counts = [1, 0, 2]
possible_neurons_per_dense_layer = [128, 64, 256]
possible_dropouts = [0.5, 0.3, 0.7]
possible_batch_sizes = [8, 4, 16]
possible_learning_rates = [0.001, 0.0005, 0.002]
possible_l2_regularizations = [0.01, 0.001, 0.1]


def suggest_config_from_trial(trial):
    """
    Ask an Optuna trial to suggest one full classifier-head config from the
    fixed grid of possible values above.

    The number of dense-layer neurons is only sampled when the trial
    suggests at least one dense layer; otherwise it falls back to
    baseline_config's value (since it is unused by `panns` in that case,
    but the config dict still needs the key).

    Args:
        trial: An `optuna.trial.Trial` (or FrozenTrial) to sample from.

    Returns:
        A `panns`-style config dict (same keys as baseline_config) with the
        sampled hyperparameters.
    """
    dense_layers = trial.suggest_categorical("dense_layers", possible_dense_layer_counts)
    neurons_per_dense_layer = trial.suggest_categorical("neurons_per_dense_layer", possible_neurons_per_dense_layer) if dense_layers > 0 else baseline_config["neurons_per_dense_layer"]
    dropout = trial.suggest_categorical("dropout", possible_dropouts)
    batch_size = trial.suggest_categorical("batch_size", possible_batch_sizes)
    learning_rate = trial.suggest_categorical("learning_rate", possible_learning_rates)
    l2_regularization = trial.suggest_categorical("l2_regularization", possible_l2_regularizations)
    return {"dense_layers": dense_layers, "neurons_per_dense_layer": neurons_per_dense_layer, "dropout": dropout, "batch_size": batch_size, "learning_rate": learning_rate, "l2_regularization": l2_regularization}


def write_trial_result(trial_number, trial_config, trial_results, completed_all_files, cutoff_reason, total_correct, total_files, total_time_seconds):
    """
    Append one trial's full outcome (config, status, per-file results) to
    results_file_path, in the plain-text format this same script's resume
    logic (see __main__) and analyze_final_best_config parse back.

    Args:
        trial_number: Optuna trial number (trial.number).
        trial_config: The `panns`-style config dict used for this trial.
        trial_results: List of per-file result dicts from `panns` (each
            with keys "file_name", "correct_species", "predicted_species",
            "predicted_confidence", "secondary_predicted_species",
            "secondary_predicted_confidence", "correct").
        completed_all_files: True if the trial ran every file in the LOOCV
            set without being cut short.
        cutoff_reason: None, or a short string explaining why the trial was
            cut short (see run_trial_loocv).
        total_correct: Number of files this trial got right.
        total_files: Total number of files in the full LOOCV set (which may
            be more than len(trial_results) if the trial was cut short).
        total_time_seconds: Wall-clock time the trial took.
    """
    with open(results_file_path, "a") as results_file:
        results_file.write(f"TRIAL: {trial_number}\n")
        for key, value in trial_config.items():
            results_file.write(f"{key}: {value}\n")
        results_file.write(f"status: {'complete' if completed_all_files else 'cut_short'}\n")
        if cutoff_reason:
            results_file.write(f"cutoff_reason: {cutoff_reason}\n")
        results_file.write(f"correct_count: {total_correct}\n")
        results_file.write(f"files_evaluated: {len(trial_results)}\n")
        results_file.write(f"total_loocv_files: {total_files}\n")
        results_file.write(f"total_time_seconds: {total_time_seconds:.1f}\n")
        results_file.write("results_per_file\n")
        for result in sorted(trial_results, key=lambda result: result["file_name"]):
            results_file.write(f"file: {result['file_name']} | correct_species: {result['correct_species']} | answer: {result['predicted_species']} | answer_confidence: {result['predicted_confidence']:.4f} | secondary_answer: {result['secondary_predicted_species']} | secondary_answer_confidence: {result['secondary_predicted_confidence']:.4f} | correct: {result['correct']}\n")
        results_file.write("\n")


def run_trial_loocv(trial, trial_config, embeddings, label_indices, file_names, species_names, files_to_test):
    """
    Run one Optuna trial's leave-one-out cross-validation: train and test
    the classifier head once per held-out file (each in its own
    short-lived worker subprocess), stopping early if this trial can no
    longer beat the best completed trial so far.

    A trial is cut short (pruned) as soon as either:
      - its running error count reaches (or exceeds) the best trial's
        final error count so far ("tied_or_exceeded_best_errors"), or
      - it has made 3 errors on files that the best trial so far got right
        ("wrong_on_3_files_best_got_right") — a heuristic for "this
        config is doing clearly worse than our best one, no point
        finishing it".
    A completed (non-cut-short) trial that beats the current best (fewer
    total errors) becomes the new best_config_so_far for subsequent trials
    to be compared against.

    Args:
        trial: The `optuna.trial.Trial` being evaluated (used for logging
            and its .number).
        trial_config: The `panns`-style config dict for this trial.
        embeddings: Precomputed PANNs embeddings for every file, as
            returned by `panns.extract_embeddings_from_folder`.
        label_indices: Integer species-label index per file.
        file_names: List of all file names in the dataset.
        species_names: List of all species names (label vocabulary).
        files_to_test: File names to run LOOCV over for this trial
            (normally the same as file_names).

    Returns:
        total_correct: Number of files this trial got right (this is also
        what gets reported back to Optuna via study.tell).
    """
    global best_config_so_far
    trial_results = []
    total_trial_errors = 0
    errors_on_files_best_got_right = 0
    cutoff_reason = None
    sorted_files = sorted(files_to_test)
    trial_start_time = time.time()
    trial_pool = multiprocessing.Pool(1, initializer=panns.initialize_worker, initargs=(trial_config, embeddings, label_indices, file_names, species_names, random_seed), maxtasksperchild=1)
    for file_index, file_name in enumerate(sorted_files, start=1):
        result = trial_pool.apply(panns.train_and_test_one_file, (file_name,))
        if result is not None:
            trial_results.append(result)
            if not result["correct"]:
                total_trial_errors += 1
                if best_config_so_far is not None and file_name in best_config_so_far["correct_files"]:
                    errors_on_files_best_got_right += 1
        print(f"[TRIAL {trial.number} PROGRESS] file {file_index}/{len(sorted_files)} - errors_so_far: {total_trial_errors}")
        if best_config_so_far is not None:
            if total_trial_errors >= best_config_so_far["errors"]:
                cutoff_reason = "tied_or_exceeded_best_errors"
                break
            if errors_on_files_best_got_right >= 3:
                cutoff_reason = "wrong_on_3_files_best_got_right"
                break
    trial_pool.close()
    trial_pool.join()
    total_time_seconds = time.time() - trial_start_time
    completed_all_files = cutoff_reason is None
    total_correct = sum(1 for result in trial_results if result["correct"])
    if completed_all_files and (best_config_so_far is None or total_trial_errors < best_config_so_far["errors"]):
        best_config_so_far = {"errors": total_trial_errors, "correct_files": set(result["file_name"] for result in trial_results if result["correct"])}
        print(f"[TRIAL {trial.number}] NEW BEST CONFIG - errors: {total_trial_errors}")
    else:
        print(f"[TRIAL {trial.number}] did not beat the current best config" + (f" - cut short: {cutoff_reason}" if cutoff_reason else ""))
    write_trial_result(trial.number, trial_config, trial_results, completed_all_files, cutoff_reason, total_correct, len(sorted_files), total_time_seconds)
    return total_correct


def analyze_final_best_config(results_file_path):
    """
    Re-read the full results file, pick the winning trial(s), and append a
    "FINAL_SELECTION" summary section to the same file.

    Only trials marked "complete" (not cut short) are considered. The
    winner is chosen by, in order: (1) most correct answers, (2) among
    ties, most additional correct answers "via secondary guess" (i.e. among
    the files this trial got wrong, how many had the correct species as the
    model's second-choice answer), (3) among further ties, lowest total
    wall-clock time. All trials tied on all three criteria are reported as
    a full tie.

    Args:
        results_file_path: Path to the results file written incrementally
            by write_trial_result.

    Returns:
        A list of winning trial dicts (more than one only in case of a
        full tie), each with keys "number", "config", "correct_count",
        "correct_via_secondary", "total_time_seconds".
    """
    with open(results_file_path, "r") as results_file:
        lines = results_file.read().split("\n")
    completed_trials = []
    line_index = 0
    while line_index < len(lines):
        if lines[line_index].startswith("TRIAL: "):
            trial_number = lines[line_index].split("TRIAL: ")[1].strip()
            line_index += 1
            trial_config = {}
            while not lines[line_index].startswith("status: "):
                key, value = lines[line_index].split(": ", 1)
                trial_config[key] = value
                line_index += 1
            status = lines[line_index].split("status: ")[1].strip()
            line_index += 1
            if status != "complete":
                line_index += 1
                continue
            total_correct = int(lines[line_index].split("correct_count: ")[1].strip())
            line_index += 2
            line_index += 1
            total_time_seconds = float(lines[line_index].split("total_time_seconds: ")[1].strip())
            line_index += 1
            while not lines[line_index].startswith("results_per_file"):
                line_index += 1
            line_index += 1
            trial_results = []
            while line_index < len(lines) and lines[line_index].startswith("file: "):
                parts = dict(part.split(": ", 1) for part in lines[line_index].split(" | "))
                trial_results.append(parts)
                line_index += 1
            wrong_files = [result for result in trial_results if result["correct"].strip() == "False"]
            correct_via_secondary = sum(1 for result in wrong_files if result["secondary_answer"].strip() == result["correct_species"].strip())
            completed_trials.append({"number": trial_number, "config": trial_config, "correct_count": total_correct, "correct_via_secondary": correct_via_secondary, "total_time_seconds": total_time_seconds})
        else:
            line_index += 1
    best_criterion = None
    winning_trials = []
    for trial in completed_trials:
        criterion = (trial["correct_count"], trial["correct_via_secondary"], -trial["total_time_seconds"])
        if best_criterion is None or criterion > best_criterion:
            best_criterion = criterion
            winning_trials = [trial]
        elif criterion == best_criterion:
            winning_trials.append(trial)
    with open(results_file_path, "a") as results_file:
        results_file.write("FINAL_SELECTION\n")
        results_file.write(f"criterion_1_correct: {winning_trials[0]['correct_count']}\n")
        results_file.write(f"criterion_2_correct_via_secondary: {winning_trials[0]['correct_via_secondary']}\n")
        if len(winning_trials) == 1:
            winning_trial = winning_trials[0]
            results_file.write(f"winning_trial: {winning_trial['number']}\n")
            results_file.write(f"total_time_seconds: {winning_trial['total_time_seconds']:.1f}\n")
            for key, value in winning_trial["config"].items():
                results_file.write(f"{key}: {value}\n")
        else:
            results_file.write(f"full_tie_among_trials: {', '.join(trial['number'] for trial in winning_trials)}\n")
            for winning_trial in winning_trials:
                results_file.write(f"--- trial {winning_trial['number']} ---\n")
                results_file.write(f"total_time_seconds: {winning_trial['total_time_seconds']:.1f}\n")
                for key, value in winning_trial["config"].items():
                    results_file.write(f"{key}: {value}\n")
    return winning_trials


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    run_is_being_resumed = os.path.exists(results_file_path) and os.path.getsize(results_file_path) > 0

    # If a results file already exists, parse every previously completed trial back out of it
    # (skipping/stopping at the first incomplete trial found, which gets re-run) so the study
    # can be resumed instead of starting over.
    already_registered_trials = []
    if run_is_being_resumed:
        with open(results_file_path, "r") as results_file:
            lines = results_file.read().split("\n")
        line_index = 0
        while line_index < len(lines):
            if lines[line_index].startswith("TRIAL: "):
                block_start_index = line_index
                try:
                    trial_number = int(lines[line_index].split("TRIAL: ")[1].strip())
                    line_index += 1
                    trial_config = {}
                    while not lines[line_index].startswith("status: "):
                        key, value = lines[line_index].split(": ", 1)
                        trial_config[key] = ast.literal_eval(value)
                        line_index += 1
                    status = lines[line_index].split("status: ")[1].strip()
                    line_index += 1
                    if lines[line_index].startswith("cutoff_reason: "):
                        line_index += 1
                    total_correct = int(lines[line_index].split("correct_count: ")[1].strip())
                    while not lines[line_index].startswith("results_per_file"):
                        line_index += 1
                    line_index += 1
                    correct_files_this_trial = set()
                    errors_this_trial = 0
                    while line_index < len(lines) and lines[line_index].startswith("file: "):
                        parts = dict(part.split(": ", 1) for part in lines[line_index].split(" | "))
                        if parts["correct"].strip() == "True":
                            correct_files_this_trial.add(parts["file"])
                        else:
                            errors_this_trial += 1
                        line_index += 1
                    already_registered_trials.append({"number": trial_number, "config": trial_config, "status": status, "correct_count": total_correct, "errors": errors_this_trial, "correct_files": correct_files_this_trial})
                except (IndexError, ValueError, KeyError):
                    print(f"incomplete trial found starting at line {block_start_index}, discarding it and re-running")
                    break
            else:
                line_index += 1
        print(f"resuming from {results_file_path} - {len(already_registered_trials)} trials already complete")
    else:
        with open(results_file_path, "w") as results_file:
            results_file.write("FIXED HYPERPARAMETERS\n")
            results_file.write(f"audio_folder: {audio_folder}\n")
            results_file.write(f"random_seed: {random_seed}\n")
            results_file.write(f"embedding_model: PANNs_CNN14\n")
            results_file.write(f"embedding_dimension: {panns.embedding_dimension}\n")
            results_file.write(f"max_epoch_cap: 200\n")
            results_file.write(f"early_stopping_patience: 10\n")
            results_file.write(f"total_trials: {total_trials}\n\n")
            results_file.write("SEARCH_SPACE\n")
            results_file.write(f"possible_dense_layer_counts: {possible_dense_layer_counts}\n")
            results_file.write(f"possible_neurons_per_dense_layer: {possible_neurons_per_dense_layer}\n")
            results_file.write(f"possible_dropouts: {possible_dropouts}\n")
            results_file.write(f"possible_batch_sizes: {possible_batch_sizes}\n")
            results_file.write(f"possible_learning_rates: {possible_learning_rates}\n")
            results_file.write(f"possible_l2_regularizations: {possible_l2_regularizations}\n\n")
            results_file.write(f"run_start_time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        print("starting a fresh run")

    audio_model = panns.AudioTagging(checkpoint_path=None, device="cpu")
    embeddings, text_labels, file_names, species_names = panns.extract_embeddings_from_folder(audio_folder, audio_model)
    label_indices = [species_names.index(label) for label in text_labels]

    best_config_so_far = None
    for registered_trial in already_registered_trials:
        if registered_trial["status"] == "complete" and (best_config_so_far is None or registered_trial["errors"] < best_config_so_far["errors"]):
            best_config_so_far = {"errors": registered_trial["errors"], "correct_files": registered_trial["correct_files"]}

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_seed))

    # Re-register every previously completed trial with the fresh Optuna study, so its TPE
    # sampler already knows about them when suggesting new configs.
    for registered_trial in already_registered_trials:
        trial_config = registered_trial["config"]
        params = {"dense_layers": trial_config["dense_layers"], "dropout": trial_config["dropout"], "batch_size": trial_config["batch_size"], "learning_rate": trial_config["learning_rate"], "l2_regularization": trial_config["l2_regularization"]}
        distributions = {"dense_layers": optuna.distributions.CategoricalDistribution(possible_dense_layer_counts), "dropout": optuna.distributions.CategoricalDistribution(possible_dropouts), "batch_size": optuna.distributions.CategoricalDistribution(possible_batch_sizes), "learning_rate": optuna.distributions.CategoricalDistribution(possible_learning_rates), "l2_regularization": optuna.distributions.CategoricalDistribution(possible_l2_regularizations)}
        if trial_config["dense_layers"] > 0:
            params["neurons_per_dense_layer"] = trial_config["neurons_per_dense_layer"]
            distributions["neurons_per_dense_layer"] = optuna.distributions.CategoricalDistribution(possible_neurons_per_dense_layer)
        study.add_trial(optuna.trial.create_trial(params=params, distributions=distributions, value=registered_trial["correct_count"]))

    if not already_registered_trials:
        study.enqueue_trial({"dense_layers": 1, "neurons_per_dense_layer": 128, "dropout": 0.5, "batch_size": 8, "learning_rate": 0.001, "l2_regularization": 0.01})

    already_tested_configs = {get_hashable_config_key(registered_trial["config"]): registered_trial["correct_count"] for registered_trial in already_registered_trials}
    completed_trial_count = len(already_registered_trials)
    while completed_trial_count < total_trials:
        trial = study.ask()
        trial_config = suggest_config_from_trial(trial)
        config_key = get_hashable_config_key(trial_config)
        if config_key in already_tested_configs:
            print(f"[TRIAL {trial.number}] repeated config, reusing result without retraining")
            study.tell(trial, already_tested_configs[config_key])
            continue
        print(f"[TRIAL {trial.number}] starting with config: {trial_config}")
        total_correct = run_trial_loocv(trial, trial_config, embeddings, label_indices, file_names, species_names, file_names)
        study.tell(trial, total_correct)
        already_tested_configs[config_key] = total_correct
        completed_trial_count += 1

    winning_trials = analyze_final_best_config(results_file_path)
    if len(winning_trials) == 1:
        print(f"BEST CONFIG FOUND - trial {winning_trials[0]['number']} - correct: {winning_trials[0]['correct_count']} - correct_via_secondary: {winning_trials[0]['correct_via_secondary']} - time: {winning_trials[0]['total_time_seconds']:.1f}s")
    else:
        print(f"FULL TIE among trials {', '.join(trial['number'] for trial in winning_trials)}")
    print(f"full results saved to {results_file_path}")