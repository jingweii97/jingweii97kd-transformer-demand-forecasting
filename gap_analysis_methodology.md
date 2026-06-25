# Methodology Gap Analysis Report

This document reports the findings of a cross-check between the research methodology described in [methodology-draft.pdf](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material%20WQF7023/repo/docs/methodology-draft.pdf) (Chapter 4) and the current Python implementation. 

We have identified **four main gaps** between the stated methodology and the active code implementation.

---

## Stated Methodology vs. Current Code Implementation

| Methodology Section / Table | Stated Requirement | Current Code Status | Gap Type | Detail & Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Section 4.3.2 (Table 4.1)** | `zero-sales indicator` is listed as an observed historical input feature. | Engineered in `data/preprocessing.py`, but omitted from the features list in `configs/dataset.yaml`. | **Feature Omission** | The models never receive the `zero_sales_indicator` as input because it is missing from `TimeSeriesDataSet`'s variable lists. |
| **Section 4.9 (Table 4.6)** | `MASE` (Mean Absolute Scaled Error) must be reported for forecasting accuracy. | Evaluates WRMSSE, MAE, RMSE, and WAPE, but completely skips MASE. | **Missing Metric** | The evaluation results CSV does not report MASE, which is a required baseline metric. |
| **Section 4.9 (Table 4.6)** | `inference time` (deployment efficiency) and `training time` (computational efficiency) must be reported. | Checkpoint sizes and parameter counts are logged, but execution/inference times are not timed or logged. | **Missing Metric** | We lack quantitative data to back up the lightweight speed claim of the student model. |
| **Section 4.9** | Forecasting performance must be reported for the overall 28-day horizon, and for short-, medium-, and long-horizon prediction ranges. | Evaluation script reports overall 28-day metrics only. | **Missing Evaluation Dimension** | We do not report sliced horizon metrics (e.g., Days 1–7, 8–14, 15–28) as requested in the methodology text. |

---

## Detailed Gap Analysis & Proposed Remedies

### Gap 1: Omission of `zero_sales_indicator` from Model Inputs
* **Methodology Statement**: Table 4.1 defines `zero-sales indicator` as an observed historical input variable used to "capture recent demand behavior... and intermittency."
* **Code Status**: In [preprocessing.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/data/preprocessing.py), the column is engineered:
  ```python
  df_long['zero_sales_indicator'] = (df_long['sales'] == 0).astype(np.int32)
  ```
  However, it is not declared under any feature category in [dataset.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/dataset.yaml).
* **Remedy**: Update [dataset.yaml](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/configs/dataset.yaml) to include `"zero_sales_indicator"` in the `time_varying_unknown_reals` (or known reals) list, allowing the PyTorch Forecasting dataset wrapper to construct it.

### Gap 2: Missing MASE Metric
* **Methodology Statement**: Table 4.6 and Section 4.9 state that forecasting accuracy is evaluated using "WRMSSE, MAE, RMSE, MASE, WAPE".
* **Code Status**: [evaluate_models.py](file:///c:/Users/jw/OneDrive%20-%20Universiti%20Malaya/Sem_2%20Study%20Material/WQF7023/repo/scripts/evaluate_models.py) implements WRMSSE, MAE, RMSE, and WAPE, but does not calculate MASE.
* **Remedy**: Integrate MASE calculation into `evaluate_models.py`. The in-sample scale factor (seasonal naive MAE denominator) can be calculated using the training data:
  $$\text{Scale}_i = \frac{1}{T - 28} \sum_{t=29}^T |Y_{i, t} - Y_{i, t-28}|$$
  Then for the evaluation window, the MASE for series $i$ is:
  $$\text{MASE}_i = \frac{\frac{1}{28}\sum_{h=1}^{28} |Y_{i, T+h} - \hat{Y}_{i, T+h}|}{\text{Scale}_i}$$

### Gap 3: Missing Training Time and Inference Time
* **Methodology Statement**: Table 4.6 lists `training time` and `inference time` as metrics to evaluate deployment and computational efficiency.
* **Code Status**: Training run durations and evaluation speeds are not tracked or saved.
* **Remedy**:
  - In `train_teacher.py` and `train_student.py`, use Python's `time.time()` to measure fit duration and save it to the experiment directory (e.g., as metadata in `config.yaml` or a dedicated metadata file).
  - In `evaluate_models.py`, measure the time taken by the prediction loop `get_predictions` and divide by the number of series/batches to log average inference speed.

### Gap 4: Missing Sliced-Horizon Reporting
* **Methodology Statement**: Section 4.9 states: "Forecasting performance is reported for the overall 28-day prediction horizon and, where applicable, for short-, medium-, and long-horizon prediction ranges..."
* **Code Status**: Evaluation outputs are flattened over the full 28-day window.
* **Remedy**: Slices of the prediction array can be evaluated separately in `evaluate_models.py`:
  - **Short-horizon**: Days 1–7 (forecast step 1 to 7)
  - **Medium-horizon**: Days 8–14 (forecast step 8 to 14)
  - **Long-horizon**: Days 15–28 (forecast step 15 to 28)
  For each slice, compute and log the corresponding accuracy metrics.
