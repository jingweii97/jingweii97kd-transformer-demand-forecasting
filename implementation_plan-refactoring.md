# Implementation Plan - Repository Refactoring and Restructuring

This plan details the steps to refactor the M5 demand forecasting codebase to conform with the [Repository Design Specification](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/docs/repository-design-specs.md). The goal is to clean separation of concerns, eliminate hardcoded values, enable multi-environment execution, and freeze the architecture.

## Aligned Decisions
From the `/grill-me` alignment, the following design specifics have been chosen:
1. **Configuration Loader**: PyYAML will load and merge YAML configs into a lightweight Python namespace object (`Config`) supporting dot-notation hierarchy (e.g. `cfg.model.hidden_size`). Environmental overrides will be handled separately from scientific hyperparameters, with built-in schema validation (type checks and missing fields) to fail fast.
2. **Dataset Cache format**: Data preprocessor in `prepare_dataset.py` will cache the cleaned, feature-engineered long-format DataFrame as an **Apache Parquet (`.parquet`)** file. PyTorch Forecasting `TimeSeriesDataSet` objects will be constructed dynamically at runtime from this Parquet file to preserve parameter configurability.
3. **Experiment Folder Convention**: Log files, CSV logs, checkpoints, and configs will be structured under `outputs/` using named experiment subfolders (e.g. `outputs/teacher/exp_001/`) rather than auto-generated Lightning version folders.

---

## Directory Restructuring Plan

We will restructure the repository into the following layout:
- `configs/` for YAML configuration files.
- `data/` for data loading, preprocessing, and caching (split boundaries are moved to configurations).
- `models/` for neural network architectures (both TFT teacher and student models).
- `utils/` for logging, paths, seed, and config validation helpers.
- `scripts/` for executable pipeline tasks.
- `artifacts/` for reusable pipeline artifacts: processed datasets under `artifacts/data/` and precomputed soft targets under `artifacts/soft_targets/` (ignored in Git).
- `outputs/` for checkpoints, logs, and evaluation reports for individual experiment runs (ignored in Git).

---

## Proposed Changes

### Configuration System

#### [NEW] [dataset.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/dataset.yaml)
Defines target name, group IDs, lookback window size ($L=90$), prediction size ($H=28$), chronological split day boundaries (Train, Validation, ID Test, OOD Test), and features list.

#### [NEW] [teacher.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/teacher.yaml)
Defines TFT teacher model hyperparameters: hidden size, attention heads, learning rate, training epochs, and batch size.

#### [NEW] [student.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/student.yaml)
Defines custom student Transformer hyperparameters: d_model, layers, heads, feed-forward dimension, dropout, learning rate, alpha weight, epochs, and batch size.

#### [NEW] [evaluation.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/evaluation.yaml)
Defines evaluation parameters and metric listings.

#### [NEW] [local.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/environment/local.yaml)
Defines local environment specifications (input_dir, artifacts_dir, outputs_dir, device/accelerator, num_workers, precision). Equivalent configurations will be created for cloud runtimes: [kaggle.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/environment/kaggle.yaml) and [dicc.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/environment/dicc.yaml).

---

### Data Processing Module (`data/`)

#### [NEW] [preprocessing.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/data/preprocessing.py)
Logic to load the raw M5 datasets, pivot sales to long format, join calendar/prices, and engineer observed lags and rolling mean/std.

#### [NEW] [dataset.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/data/dataset.py)
Logic to instantiate the PyTorch Forecasting `TimeSeriesDataSet` from a DataFrame using split day ranges and configurations from the loaded `Config` object.

#### [NEW] [cache.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/data/cache.py)
Saves/loads preprocessed DataFrames to/from Parquet files under `artifacts/data/`.

---

### Neural Network Models (`models/`)

#### [NEW] [teacher.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/models/teacher.py)
A wrapper/factory module to instantiate the PyTorch Forecasting `TemporalFusionTransformer` teacher model from the system configuration.

#### [NEW] [student.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/models/student.py)
Defines the `M5TransformerStudent` PyTorch LightningModule (embedding categorical variables, sequence projection, and joint Huber distillation loss logic).

---

### Common Utilities (`utils/`)

#### [NEW] [config.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/utils/config.py)
Loads the YAML configuration files, merges them (experiment config + environment config), handles schema/type validation, exposes them as a dot-notation Python object, and supports saving the fully merged config to a specific experiment folder for complete reproducibility.

#### [NEW] [paths.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/utils/paths.py)
Resolves environmental paths relative to the Git repository root directory.

#### [NEW] [logging.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/utils/logging.py)
Initializes PyTorch Lightning `CSVLogger` under target named experiment folders.

#### [NEW] [seed.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/utils/seed.py)
Sets global seeds for reproducibility.

---

### Executable Scripts (`scripts/`)

#### [NEW] [prepare_dataset.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/scripts/prepare_dataset.py)
Loads configurations, preprocesses raw data, and writes the Parquet file to `artifacts/data/`.

#### [NEW] [train_teacher.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/scripts/train_teacher.py)
Trains the TFT teacher model on the training window, early-stops on validation, and writes the best model checkpoint and config metadata to `outputs/teacher/<exp_name>/`.

#### [NEW] [generate_soft_targets.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/scripts/generate_soft_targets.py)
Generates predictions from a frozen teacher checkpoint over all sliding training windows and writes the 3D soft targets tensor to `artifacts/soft_targets/<exp_name>.pt`.

#### [NEW] [train_student.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/scripts/train_student.py)
Trains student Transformer models (with or without KD) and writes checkpoints and configuration metadata to `outputs/student/<exp_name>/`.

#### [NEW] [evaluate_models.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/scripts/evaluate_models.py)
Loads checkpoints, runs prediction inference on ID and OOD splits, computes WRMSSE across 12 hierarchy levels, and saves evaluation CSV outputs and config metadata to `outputs/evaluation/<exp_name>/`.

---

## Cleanup Plan
Once the new structure is built and verified, we will:
1. Delete the legacy files: `scripts/data_utils.py`, `scripts/student_model.py`, and any old checkpoints/cached files directly in the root or `input/` folder (except raw CSVs).
2. Update the Git ignore definitions (`.gitignore`) to exclude `artifacts/`, `outputs/`, and raw data to avoid committing heavy files.

---

## Verification Plan
We will verify the restructured codebase by running a **Phase 1 fast validation check on store `CA_1`**:
1. Run `prepare_dataset.py` with `store_filter: "CA_1"` to generate Parquet file in `artifacts/data/`.
2. Run `train_teacher.py` for 1 epoch (`limit_train_batches: 10`, `limit_val_batches: 5`) to confirm checkpoint and config metadata writing to `outputs/teacher/`.
3. Run `generate_soft_targets.py` to generate the 3D tensor lookup in `artifacts/soft_targets/`.
4. Run `train_student.py` (with KD and without KD) for 1 epoch to verify gradient flow, checkpointing, and config metadata writing to `outputs/student/`.
5. Run `evaluate_models.py` to confirm the metrics CSV and config metadata are written to `outputs/evaluation/`.
