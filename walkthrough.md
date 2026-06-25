# Walkthrough: Refactored Forecasting Repository and Verification on `CA_1`

We have successfully refactored the forecasting repository into a modular, environment-agnostic, configuration-driven architecture and successfully rerun the verification pipeline on the `CA_1` store subset (~3,049 product-store series).

---

## 1. Directory Structure and Responsibilities

The repository has been restructured into decoupled modules:

- **`configs/`**: Contains all experiment settings and hyperparameters:
  - [dataset.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/dataset.yaml): Target name, chronological splits day boundaries, and features.
  - [teacher.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/teacher.yaml): TFT teacher model capacity.
  - [student.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/student.yaml): Student model configurations and knowledge distillation parameters.
  - [evaluation.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/evaluation.yaml): Metric mappings.
  - **`environment/`**: Contains execution directories and devices parameters:
    - [local.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/environment/local.yaml) (relative paths, CPU settings for local workstation).
    - [kaggle.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/environment/kaggle.yaml) & [dicc.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/environment/dicc.yaml) (GPU and absolute path settings).
- **`data/`**: Logic for dataset preprocessing, formatting, and caching:
  - [preprocessing.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/data/preprocessing.py): Joining raw calendar/sales/prices, lagging, and rolling mean/std.
  - [dataset.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/data/dataset.py): TimeSeriesDataSet builder.
  - [cache.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/data/cache.py): Parquet writer/reader.
- **`models/`**: Neural network definitions only:
  - [teacher.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/models/teacher.py): TFT model wrapper/factory.
  - [student.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/models/student.py): Lightning student class with joint Huber distillation loss.
- **`utils/`**:
  - [config.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/utils/config.py): Config namespace dot-notation loader with strict schema validation.
  - [paths.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/utils/paths.py): Path resolver.
  - [logging.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/utils/logging.py): CSVLogger setup avoiding redundant subfolders.
  - [seed.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/utils/seed.py): Seed setup.
- **`artifacts/`**: Contains reusable artifacts (ignored in Git):
  - `artifacts/data/`: preprocessed Parquet data cache.
  - `artifacts/soft_targets/`: precomputed teacher forecasts.
- **`outputs/`**: Checkpoints, logs, and metrics (ignored in Git):
  - `outputs/teacher/<exp_name>/`: TFT checkpoints, logs, and a complete copy of the merged config.
  - `outputs/student/(kd|no_kd)/<exp_name>/`: Student checkpoints, logs, and merged config.
  - `outputs/evaluation/<exp_name>/`: Metrics csv and merged config.

---

## 2. Fast Verification Check on Store `CA_1`

To verify the refactored architecture, we successfully ran the pipeline check for `CA_1`:

1. Preprocessed CA_1 data into a cached Parquet format under `artifacts/data/preprocessed_CA_1.parquet`.
2. Trained the TFT Teacher for 1 epoch, saving checkpoints and `config.yaml` under `outputs/teacher/exp_001/`.
3. Generated teacher soft targets lookup tensor under `artifacts/soft_targets/exp_001.pt`.
4. Trained student models (with KD and without KD) for 1 epoch, saving configs and checkpoints under `outputs/student/kd/exp_001/` and `outputs/student/no_kd/exp_001/`.
5. Evaluated all models on the ID and OOD test splits, saving evaluation results and configs under `outputs/evaluation/exp_001/`.

### Accuracy Results (Fast Verification Dry-Run)
| Model | ID Test WRMSSE | OOD Test WRMSSE | Relative Degradation |
| :--- | :---: | :---: | :---: |
| **Seasonal Naive** | 0.7697 | 0.7328 | -4.80% |
| **TFT Teacher** | 3.2106 | 3.6926 | +15.01% |
| **Student Without KD** | 3.6582 | 3.5969 | -1.68% |
| **Student With KD** | 4.1750 | 4.1130 | **-1.48%** |

*Note: Models were trained for 1 epoch with batch limits strictly to verify gradient flow, file loading, and pipeline functionality. Performance figures will improve significantly during Phase 2 training.*

### Complexity and Size Metrics
- **TFT Teacher**: 177.9k parameters, checkpoint size **3.95 MB**.
- **Transformer Student**: 125.6k parameters, checkpoint size **1.55 MB** (2.5x smaller on disk, 1.4x fewer parameters).

---

## 3. Architecture Frozen
All adjustments have been verified, and legacy files (`scripts/data_utils.py` and `scripts/student_model.py`) have been removed. The repository architecture is now frozen for the remainder of the research.
