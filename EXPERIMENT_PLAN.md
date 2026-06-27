# Experiment Plan

## Project

**Title:** Lightweight Transformer Models for Multi-Horizon Demand Forecasting under Distribution Shifts in Retail Transaction Data

**Purpose:** This document defines the internal experimental plan for the full M5 experimental phase. It separates implementation/debugging work from reported scientific experiments and fixes the experimental sequence, model-selection criteria, tuning scope, evaluation metrics, and artifact structure before full-scale training begins.

---

## 1. Purpose and Scope

Phase 2A has verified that the repository pipeline is functional. The CA_1 walkthrough confirms that data loading, preprocessing, chronological splitting, sample construction, teacher training, student training, knowledge distillation, evaluation, and artifact saving can run successfully.

However, CA_1 is treated as **engineering validation only**. It is used to verify that the software pipeline works, not to provide final scientific evidence. All reported hyperparameter tuning, model selection, final training, and final evaluation will be conducted using the **full M5 dataset** and the predefined chronological splits in the methodology chapter.

CA_1 may still be used for debugging, smoke testing, environment checks, or verifying that code changes do not break the pipeline. CA_1 results should not be reported as final thesis results.

---

## 2. Fixed Methodological Settings

The following settings are fixed before full-scale experiments.

| Component | Fixed setting |
|---|---|
| Dataset | Full M5 Walmart dataset |
| Training window | `d_1` to `d_1857` |
| Validation window | `d_1858` to `d_1885` |
| ID test window | `d_1886` to `d_1913` |
| OOD test window | `d_1914` to `d_1941` |
| Forecast horizon | `H = 28` days |
| Lookback window | `L = 90` days |
| Sliding-window stride | 1 day |
| Teacher model family | Temporal Fusion Transformer |
| Student model family | Compact Transformer student |
| Knowledge distillation type | Output-level knowledge distillation |
| Tuning data | Validation window only |
| Final evaluation data | ID and OOD test windows only |

The ID and OOD test windows are reserved exclusively for final evaluation and must not be used for hyperparameter tuning or model selection.

---

## 3. Model Groups

The final experimental comparison includes four model groups.

| Model | Role | Purpose |
|---|---|---|
| Seasonal naïve | Statistical baseline | Provides a simple forecasting benchmark |
| TFT teacher | High-capacity reference model | Provides teacher forecasts and upper-performance reference |
| Compact Transformer student without KD | Ablation baseline | Measures compact student performance without teacher guidance |
| Compact Transformer student with KD | Proposed model | Evaluates whether KD improves lightweight forecasting |

---

## 4. Evaluation Criteria

The experiments are evaluated using three main criteria.

### 4.1 Forecasting Accuracy

Forecasting accuracy is assessed using:

- WRMSSE;
- MAE;
- RMSE;
- MASE;
- WAPE.

Validation WRMSSE is the primary model-selection metric. If WRMSSE is too expensive to compute at every epoch, validation loss may be used for early stopping, but final configuration selection should be based primarily on validation WRMSSE.

### 4.2 Deployment Efficiency

Deployment efficiency is assessed using:

- trainable parameter count;
- saved model size;
- inference time under a controlled evaluation environment.

Training time is not used as a formal lightweight metric because it is sensitive to hardware, data loading, software environment, and runtime conditions.

The compact Transformer student is expected to demonstrate meaningful deployment efficiency improvement relative to the TFT teacher. Actual reductions in parameter count, saved model size, and inference time will be reported and interpreted based on the final experimental results.

### 4.3 Temporal Stability

Temporal stability is assessed using relative ID-to-OOD degradation:

```text
Relative Degradation (%) = ((OOD Error - ID Error) / ID Error) × 100
```

The same error metric should be used consistently when reporting degradation. WRMSSE is preferred as the primary degradation metric.

---

## 5. Loss Functions for Student Training

The student with knowledge distillation uses a combined supervised and distillation loss:

```text
L_total = alpha * L_supervised + (1 - alpha) * L_distillation
```

Recommended implementation:

- `L_supervised`: Huber Loss / Smooth L1 between student forecasts and ground-truth demand;
- `L_distillation`: Mean Squared Error between student forecasts and TFT teacher forecasts.

The distillation weighting coefficient `alpha` is tuned using the validation window.

---

## 6. Experimental Sequence

The full experimental phase is conducted in five experiments.

| Experiment ID | Experiment | Main purpose | Output |
|---|---|---|---|
| EXP-001 | TFT teacher tuning | Select teacher configuration | Best teacher configuration and checkpoint |
| EXP-002 | Student architecture tuning | Select compact student configuration | Final student configuration |
| EXP-003 | KD tuning | Select distillation coefficient `alpha` | Final KD configuration |
| EXP-004 | Final training | Train final model set using fixed configurations | Final checkpoints |
| EXP-005 | Final evaluation | Evaluate final models on ID and OOD windows | Chapter 5 tables and figures |

---

## 7. EXP-001: TFT Teacher Tuning

### Purpose

Select a strong TFT teacher configuration using the full M5 training and validation windows. The teacher should provide a meaningful high-capacity reference for knowledge distillation.

### Initial Candidate Search Space

The search space below is an initial bounded search space. Values may be adjusted if hardware constraints or validation results indicate that a minor revision is necessary. Any changes must be logged.

| Hyperparameter | Initial candidate values |
|---|---|
| Learning rate | `1e-3`, `5e-4` |
| Hidden size | `64`, `128` |
| Hidden continuous size | `16`, `32` |
| Attention head size | `2`, `4` |
| LSTM layers | `1`, `2` |
| Dropout | `0.1`, `0.2` |

### Selection Criterion

Select the teacher configuration primarily using validation WRMSSE. If two configurations perform similarly, prefer the configuration with more stable validation behaviour and reasonable deployment cost.

### Expected Outputs

- best teacher configuration;
- best teacher checkpoint;
- validation metrics;
- model metadata.

---

## 8. EXP-002: Compact Student Architecture Tuning

### Purpose

Select a compact Transformer student configuration that balances forecasting accuracy and deployment efficiency.

### Initial Candidate Search Space

The search space below is an initial bounded search space. Values may be adjusted if required, but all changes must be documented.

| Hyperparameter | Initial candidate values |
|---|---|
| Learning rate | `1e-3`, `5e-4` |
| Transformer encoder layers | `1`, `2` |
| Model dimension / hidden dimension | `32`, `64` |
| Attention heads | `1`, `2` |
| Feed-forward dimension | `64`, `128` |
| Dropout | `0.1`, `0.2` |

### Selection Criterion

Select the student configuration using:

1. validation WRMSSE as the primary criterion;
2. trainable parameter count, saved model size, and inference time as deployment-efficiency criteria.

The final selected student architecture must be used unchanged for both the student without KD and the student with KD. This ensures that the effect of KD is isolated from architectural differences.

### Expected Outputs

- final student configuration;
- validation metrics for student without KD;
- deployment-efficiency metadata.

---

## 9. EXP-003: Knowledge Distillation Tuning

### Purpose

Select the distillation weighting coefficient `alpha` for the compact Transformer student with KD.

### Initial Candidate Values

```text
alpha ∈ {0.3, 0.5, 0.7}
```

If time and compute resources permit, `alpha = 0.9` may also be tested.

### Selection Criterion

Select `alpha` primarily using validation WRMSSE for the student with KD. Secondary considerations include improvement over the student without KD and validation-loss stability.

### Expected Outputs

- KD tuning results;
- final KD configuration;
- validation metrics for student with KD.

---

## 10. EXP-004: Final Training

### Purpose

Train the final model set using fixed configurations selected from EXP-001 to EXP-003.

### Final Models

The final model set includes:

1. Seasonal naïve baseline;
2. TFT teacher;
3. Compact Transformer student without KD;
4. Compact Transformer student with KD.

### Rules

After final configurations are frozen:

- no additional hyperparameter changes should be made;
- no ID or OOD test metrics should be used for model selection;
- checkpoints and results should be saved with clear experiment identifiers.

### Expected Outputs

- final configurations;
- final checkpoints where applicable;
- training logs;
- model metadata.

---

## 11. EXP-005: Final Evaluation

### Purpose

Evaluate the final model set on the predefined ID and OOD test windows.

### Evaluation Outputs

The final evaluation should produce:

- forecasting accuracy results;
- deployment efficiency results;
- ID/OOD degradation results;
- accuracy-efficiency trade-off analysis;
- plots and tables for Chapter 5.

### Interpretation Focus

The final results should answer the following questions:

1. Does the student with KD improve over the student without KD?
2. How much forecasting performance does the student with KD retain relative to the TFT teacher?
3. What deployment-efficiency improvement does the compact student achieve relative to the TFT teacher?
4. How do the models differ in ID-to-OOD degradation?

The performance gap between the KD student and the TFT teacher should be analysed in relation to the deployment efficiency gained. No fixed percentage threshold is imposed before experimentation; actual performance retention and efficiency trade-offs will be interpreted based on the final results.

---

## 12. Experiment Logging Convention

Each experiment should use a consistent experiment identifier.

Recommended naming pattern:

```text
EXP###_<stage>_<key-settings>
```

Examples:

```text
EXP001_teacher_h128_lstm2_lr5e4_dp01
EXP002_student_d64_l2_h2_ff128_lr1e3
EXP003_kd_alpha07
EXP004_final_student_kd
EXP005_eval_final_models
```

Each experiment folder should include metadata documenting:

- experiment ID;
- git commit or repository tag;
- dataset setting;
- temporal split setting;
- configuration file;
- random seed;
- hardware environment;
- start and end time;
- notes on any deviations from the plan.

---

## 13. Artifact Organization

Recommended output structure:

```text
outputs/
  phase2_full_m5/
    exp_001_teacher_tuning/
    exp_002_student_tuning/
    exp_003_kd_tuning/
    exp_004_final_training/
    exp_005_final_evaluation/
```

Each experiment folder should contain the relevant configuration files, logs, checkpoints, validation metrics, evaluation outputs, plots, and metadata.

---

## 14. Version Control

Before starting full experiments:

1. merge Phase 2A changes;
2. tag the repository;
3. record the tag or commit hash in this document.

Recommended tag:

```text
phase2a-complete
```

or:

```text
v0.2-methodology-aligned
```

All full M5 experiments should reference this tag or a later documented experiment tag.

---

## 15. Reporting Plan

Chapter 5 should report only the controlled full M5 experiments.

Do not report CA_1 debugging results as final model evidence.

Report final results using:

1. forecasting accuracy table;
2. deployment efficiency table;
3. ID/OOD degradation table;
4. accuracy-efficiency trade-off figure;
5. KD comparison discussion.

---

## Appendix A: Kaggle Execution Checklist

Before running full experiments on Kaggle:

1. Create a clean Kaggle notebook.
2. Clone or upload the repository.
3. Attach the M5 dataset.
4. Install required packages.
5. Confirm GPU availability.
6. Confirm repository paths and configuration loading.
7. Confirm output directories are writable.
8. Run a one-epoch smoke test.
9. Confirm checkpoint saving.
10. Confirm evaluation metric computation.
11. Confirm artifacts are saved and downloadable.

Do not start full training until the smoke test passes.

---

## Appendix B: Kaggle Smoke Test

The smoke test should use a small configuration and limited training run.

Recommended smoke test settings:

- 1 epoch;
- small batch limit or reduced number of batches;
- checkpoint saving enabled;
- evaluation enabled;
- artifact saving enabled.

The smoke test is considered successful if:

- the repository executes in the Kaggle environment;
- GPU is detected;
- model training starts and completes one epoch;
- checkpoint is saved;
- predictions are generated;
- evaluation metrics are computed;
- output artifacts are written correctly.

Smoke test results are not reported as thesis results.

---

## Appendix C: Final Pre-Experiment Checklist

Before launching EXP-001, confirm the following:

- repository is frozen and tagged;
- experiment plan has been reviewed;
- full M5 dataset path is correct;
- chronological splits match the methodology;
- `L = 90`, `H = 28`, and stride = 1 are configured;
- validation WRMSSE is available for model selection;
- output folders are configured;
- Kaggle smoke test has passed;
- search spaces are bounded;
- final metrics are defined;
- no ID/OOD test metrics are used for tuning.

---

## Summary

This experiment plan defines the transition from engineering validation to controlled scientific experimentation. CA_1 verifies that the software pipeline works, while the full M5 experiments provide the evidence required for Chapter 5. The plan fixes the experimental sequence, model-selection criteria, tuning scope, evaluation metrics, logging convention, and artifact structure before full-scale training begins.
