# Focused EDA Context for ID/OOD Temporal Scenario Design (M5 Forecasting)

## Objective

The purpose of this EDA is NOT general exploratory analysis for the entire thesis.

The purpose is specifically to:

1. Characterize temporal demand behaviour in the M5 dataset.
2. Identify and justify fixed ID and OOD evaluation windows.
3. Support the methodology section for temporal distribution shift evaluation.
4. Select evaluation windows BEFORE model training.

The EDA should focus on temporal regime characterization rather than forecasting performance.

---

# Research Context

The thesis evaluates lightweight multi-horizon retail demand forecasting under temporally shifted conditions.

The forecasting evaluation protocol includes:

* Training period
* Validation period
* ID (in-distribution) temporal test window
* OOD (temporally shifted) test window

The OOD scenario should represent a retail period with measurably different temporal characteristics compared with the training and ID periods.

The OOD scenario must:

* come from real M5 temporal behaviour,
* be fixed before model training,
* and NOT be selected based on forecasting performance.

---

# Dataset Components

Use the standard M5 dataset components:

## Required files

* sales_train_validation.csv
* calendar.csv
* sell_prices.csv

Optional:

* sample_submission.csv (not required for EDA)

---

# Main EDA Goal

Identify suitable 28-day windows for:

* ID evaluation
* OOD evaluation

The ID window should:

* represent a relatively ordinary near-future retail period.

The OOD window should:

* represent a temporally shifted retail regime,
* such as higher event intensity,
* stronger volatility,
* increased price variation,
* increased intermittency,
* or unusual demand behaviour.

---

# Chronological Constraints

All temporal windows must:

* preserve chronological order,
* avoid leakage,
* and remain fixed before model training.

Example structure:

* Training: earlier historical period
* Validation: next chronological window
* ID: near-future ordinary window
* OOD: later temporally shifted window

Do NOT randomize temporal splits.

---

# Required EDA Outputs

## 1. Rolling 28-Day Temporal Statistics

Compute rolling or non-overlapping 28-day statistics over time.

Required indicators:

### Demand statistics

* mean sales
* median sales
* sales standard deviation
* coefficient of variation
* demand volatility

### Intermittency statistics

* zero-sales proportion
* active SKU proportion

### Event/calendar statistics

* event frequency
* SNAP frequency
* holiday count

### Price statistics

* average price variation
* price-change frequency

---

# 2. Candidate Window Characterization

Generate summaries for candidate 28-day windows.

For each window:

* start day
* end day
* mean sales
* volatility
* zero-sales ratio
* event density
* SNAP density
* price variation

Create a comparison table between:

* training window,
* candidate ID windows,
* candidate OOD windows.

---

# 3. Temporal Visualization

Generate visualizations such as:

## Time-series plots

* total sales over time
* rolling volatility
* rolling zero-sales ratio
* event density over time
* SNAP activity over time

## Window comparison plots

* boxplots or violin plots of sales distributions
* volatility comparison
* price-change comparison

---

# 4. Distribution Difference Indicators

Compute simple descriptive shift indicators between:

* training vs ID,
* training vs OOD.

Possible indicators:

* mean difference
* variance difference
* KS statistic
* Wasserstein distance (optional)
* relative volatility difference

IMPORTANT:
These indicators are descriptive only.
Do NOT optimize windows using model performance.

---

# OOD Selection Strategy

The OOD window should be selected using:

* predefined temporal indicators,
* not forecasting accuracy.

The selected OOD window should:

* occur after the ID window chronologically,
* show stronger temporal shift characteristics,
* remain fixed before training.

Possible OOD candidates:

* December holiday periods
* event-heavy periods
* high-volatility retail windows

---

# Expected Final Deliverables

The EDA should produce:

## 1. Final selected windows

Example:

* Training: d_x to d_y
* Validation: d_y+1 to d_z
* ID: d_a to d_b
* OOD: d_c to d_d

## 2. Quantitative justification

Why the OOD window differs from the ID/training periods.

## 3. Figures and tables

Suitable for inclusion in thesis methodology/results chapters.

---

# Important Constraints

This EDA is NOT:

* forecasting model training,
* drift detection,
* adaptive learning,
* or anomaly detection.

This EDA ONLY supports:

* temporal scenario definition,
* methodology grounding,
* and evaluation design.

---

# Downstream Methodology Usage

The selected windows will later be used for:

1. TFT teacher evaluation
2. Compact student without KD
3. Compact student with KD
4. Seasonal naive baseline

Performance comparison will include:

* ID forecasting error
* OOD forecasting error
* relative degradation

Relative degradation formula:

Relative degradation (%) =
(OOD error - ID error) / ID error × 100

---

# Implementation Notes

Preferred stack:

* Python
* pandas
* numpy
* matplotlib
* seaborn (optional)
* scipy (optional)

Code should:

* be modular,
* reproducible,
* and configurable for different candidate windows.
