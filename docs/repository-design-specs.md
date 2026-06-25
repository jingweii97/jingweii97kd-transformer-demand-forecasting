# Repository Design Specification

**Project:** Lightweight Transformer with Knowledge Distillation for Multi-Horizon Demand Forecasting under Distribution Shift

**Version:** 1.0

**Status:** Final Architecture (Freeze After Refactoring)

---

# 1. Purpose

This document defines the final software architecture for the research repository.

The objective is to ensure that:

* experiments are reproducible;
* local and cloud execution use the same codebase;
* every experiment is traceable;
* no major architectural refactoring is required after Phase 1.

After implementation of this specification, repository restructuring is considered complete. Future work should focus exclusively on research experiments, bug fixes, and thesis writing.

---

# 2. Design Principles

The repository follows seven engineering principles.

## Principle 1 — Configuration Driven

All experiment settings must come from configuration files.

Python modules should contain implementation logic only.

Configurations include:

* dataset
* model
* training
* evaluation
* environment

No model hyperparameters should be hardcoded inside Python source files.

---

## Principle 2 — Environment Agnostic

The same repository must execute on:

* Local Windows workstation
* Kaggle
* Google Colab
* UM DICC
* Linux GPU server

No code modifications should be necessary when changing execution environment.

Only configuration files should differ.

---

## Principle 3 — One Script = One Responsibility

Each executable script performs exactly one task.

Scripts should never perform multiple independent stages.

Example:

```
prepare_dataset.py

↓

train_teacher.py

↓

generate_soft_targets.py

↓

train_student.py

↓

evaluate_models.py
```

---

## Principle 4 — Reproducibility

Every experiment must be reproducible.

Each experiment must automatically record:

* configuration
* random seed
* timestamp
* checkpoint
* metrics

---

## Principle 5 — Artifact-Based Workflow

Expensive computations are performed only once.

Reusable artifacts include:

* processed datasets
* teacher checkpoints
* soft targets
* evaluation outputs

Training scripts should consume artifacts instead of regenerating them.

---

## Principle 6 — Research First

Repository design should support research.

Avoid unnecessary software engineering complexity.

The repository is not intended to become a production ML system.

---

## Principle 7 — Architecture Freeze

After implementation of this specification:

* no directory restructuring
* no module relocation
* no framework migration

Only:

* experiments
* bug fixes
* thesis improvements

are allowed.

---

# 3. Repository Layout

```
repo/

├── configs/
│
├── data/
│
├── docs/
│
├── input/
│
├── models/
│
├── outputs/
│
├── scripts/
│
├── utils/
│
├── notebooks/
│
└── README.md
```

---

# 4. Directory Responsibilities

## configs/

Contains all experiment configuration.

```
configs/

dataset.yaml

teacher.yaml

student.yaml

evaluation.yaml

environment/

    local.yaml

    kaggle.yaml

    dicc.yaml
```

Responsibilities

* dataset paths
* hyperparameters
* optimizer settings
* batch sizes
* hardware settings

---

## data/

Contains reusable data processing modules.

```
data/

preprocessing.py

dataset.py

splits.py

cache.py
```

Responsibilities

* preprocessing
* feature engineering
* chronological split
* dataset construction
* caching

No model training.

---

## models/

Contains neural network definitions only.

```
models/

student.py
```

Responsibilities

* architecture
* forward pass
* losses

No training loops.

---

## scripts/

Executable entry points.

```
prepare_dataset.py

train_teacher.py

generate_soft_targets.py

train_student.py

evaluate_models.py
```

Each script performs one task only.

---

## utils/

Common utilities.

```
config.py

device.py

logging.py

paths.py

seed.py
```

Responsibilities

* configuration loading
* device selection
* logger creation
* random seed
* common helper functions

---

## outputs/

Contains every experiment result.

```
outputs/

teacher/

student/

evaluation/

soft_targets/
```

Nothing inside this directory should be committed to Git.

---

## notebooks/

Used only for:

* exploratory data analysis
* visualization
* result inspection

Never used for training.

---

# 5. Configuration System

Configurations are divided into two categories.

## Experiment Configuration

Defines the scientific experiment.

Example:

```
teacher.yaml

student.yaml

evaluation.yaml
```

Contains:

* hidden size
* attention heads
* layers
* learning rate
* alpha
* epochs
* lookback
* forecast horizon

These settings should never change because of hardware.

---

## Environment Configuration

Defines execution environment.

Example:

```
local.yaml

kaggle.yaml

dicc.yaml
```

Contains:

* input directory
* output directory
* device
* num_workers
* precision
* accelerator

Environment files must never change scientific hyperparameters.

---

# 6. Output Convention

Every experiment receives its own directory.

Example

```
outputs/

teacher/

    exp001/

        config.yaml

        metrics.csv

        model.ckpt

        loss.png

student/

    kd/

        exp001/

    no_kd/

        exp001/

evaluation/

    exp001/

soft_targets/

    teacher_exp001.pt
```

Every experiment must be independently reproducible.

---

# 7. Logging

Training logs should include:

* training loss
* validation loss
* learning rate
* epoch
* elapsed time

Experiment folder should automatically save:

* configuration
* metrics
* checkpoint

Avoid anonymous directories such as

```
version_0

version_1

version_2
```

Prefer meaningful experiment names.

---

# 8. Cloud Workflow

Development occurs locally.

```
Develop

↓

Verify on CA_1

↓

Commit to Git

↓

Push

↓

Pull on Cloud

↓

Run Full M5

↓

Download Results
```

Cloud should execute the identical codebase.

Only environment configuration changes.

---

# 9. Coding Standards

* Relative paths only
* No duplicated logic
* No hardcoded hyperparameters
* No notebook-only functionality
* Type hints where practical
* Meaningful variable names
* Docstrings for public functions

---

# 10. Experiment Lifecycle

Every experiment follows the same sequence.

```
1.

Prepare Dataset

↓

2.

Train Teacher

↓

3.

Generate Soft Targets

↓

4.

Train Student

↓

5.

Evaluate

↓

6.

Archive Results
```

No stage should regenerate outputs from previous stages.

---

# 11. Version Control

Git tracks:

* source code
* configurations
* documentation

Git does NOT track:

* checkpoints
* datasets
* logs
* outputs
* cached files

Use `.gitignore` accordingly.

---

# 12. Architecture Freeze Policy

Once this specification is implemented:

* Repository structure is frozen.
* Future changes should not move files or redesign modules.
* New research ideas (e.g., Diffusion Teacher, new KD losses, alternative student architectures) must integrate into the existing structure rather than reorganizing it.

This policy minimizes regression risk and ensures all subsequent effort is directed toward scientific experimentation and thesis completion.

---

# Final Goal

The repository should behave like a reproducible research project rather than a collection of experimental scripts.

A new researcher should be able to:

1. Clone the repository.
2. Install dependencies.
3. Select an environment configuration.
4. Run the pipeline.
5. Reproduce every experiment reported in the thesis without modifying the source code.
