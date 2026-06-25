# Research Context and Methodology Decisions: M5 ID/OOD forecasting with KD

This document serves as the globally agreed context and decision log for implementing the research methodology outlined in the thesis chapter.

---

## 1. Research Objective & Scope
The objective is to implement and evaluate a lightweight Transformer-based student model for 28-day multi-horizon retail demand forecasting on the M5 Walmart dataset under temporally shifted conditions. The study investigates whether **output-level knowledge distillation (KD)** enables a compact student to retain the forecasting performance of a high-capacity **Temporal Fusion Transformer (TFT)** teacher model while reducing deployment costs.

---

## 2. Experimental Data & Chronological Split
The evaluation uses the M5 Walmart dataset, with the timeline split chronologically into four non-overlapping windows:

| Split / Window | Day Range | Calendar Dates | Duration (Days) | Purpose / Description |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | Day 1 – Day 1,857 | 2011-01-29 to 2016-02-28 | 1,857 days | Model training |
| **Validation** | Day 1,858 – Day 1,885 | 2016-02-29 to 2016-03-27 | 28 days | Hyperparameter tuning, early stopping, and model selection only (not used for final evaluation) |
| **ID Test** | Day 1,886 – Day 1,913 | 2016-03-28 to 2016-04-24 | 28 days | Predefined near-future in-distribution evaluation (reported results) |
| **OOD Test** | Day 1,914 – Day 1,941 | 2016-04-25 to 2016-05-22 | 28 days | Predefined temporally shifted out-of-distribution evaluation (reported results) |

*Note: The Validation window is used strictly for model selection, hyperparameter tuning, and early stopping. This tuning process occurs exclusively on the full M5 dataset during the experimental phase, and the Validation split is excluded from all final performance reports. The final thesis performance comparisons are evaluated and reported exclusively on the predefined ID and OOD test windows. Temporal distribution shift is treated as an evaluation concern, not a drift-adaptation problem. The OOD window is characterized by higher event intensity (2 holidays, event frequency 0.14 vs 0.0) and greater price volatility compared to the ID window.*

---

## 3. Model Inputs & Features
Features are grouped into four categories, matching the TFT and Transformer student structures:

1. **Target Series**: Daily unit sales (prediction target).
2. **Observed Historical Inputs (Lookback window $L=90$ days)**:
   - Historical sales sequence
   - Lags (7-day lag, 28-day lag)
   - Rolling statistics (7-day rolling mean, 7-day rolling standard deviation)
   - Zero-sales indicator
3. **Known Future Inputs (Forecast horizon $H=28$ days)**:
   - Calendar variables (Day of week, month)
   - Event indicators (Event name, event type, holiday counts)
   - SNAP indicators (CA, TX, WI)
   - Price variables (Selling price, price-change indicator, percentage price change)
4. **Static Categorical Inputs (Product-store hierarchy)**:
   - Item ID, Department ID, Category ID, Store ID, State ID

---

## 4. Models and Roles

1. **Seasonal Naïve (Statistical Baseline)**:
   - Simple baseline for verifying forecasting capability.
2. **TFT Teacher (High-Capacity Reference Model)**:
   - High-capacity reference model trained directly on ground truth. Provides soft targets for distillation.
3. **Compact Transformer Student (Without KD)**:
   - Ablation baseline sharing the student architecture but trained purely on ground-truth labels.
4. **Compact Transformer Student (With KD)**:
   - Proposed model sharing the student architecture but trained on a blended loss of ground truth and teacher soft targets.

---

## 5. Knowledge Distillation Strategy
- **Type**: Output-level knowledge distillation (hidden-state, attention-map, and variable-selection distillation are excluded).
- **Objective Function**: 
  $$L_{total} = \alpha L_{supervised} + (1 - \alpha) L_{distillation}$$
  where:
  - $L_{supervised}$ is the loss (e.g., MSE or Hubers loss) between the student's forecasts and ground-truth values.
  - $L_{distillation}$ is the loss between the student's forecasts and the frozen teacher's forecasts.
  - $\alpha \in [0, 1]$ is a tunable hyperparameter.

---

## 6. Aligned Implementation Decisions

The following details have been finalized and aligned:
- **Framework & Modeling Library**: **PyTorch Forecasting** (built on PyTorch Lightning), leveraging its native, tested `TemporalFusionTransformer` and `TimeSeriesDataSet` pipeline.
- **Dataset Scale & Strategy**: A two-phase implementation strategy will be followed:
  - **Phase 1 (Implementation and Development)**: Uses strictly the **CA_1 store subset** (~3,049 product-store series) for software development, debugging, and pipeline verification. **No scientific conclusions or hyperparameter tuning are conducted on this subset.**
  - **Phase 2 (Experimental Phase)**: Conducted exclusively on the **full M5 dataset** (all 30,490 series). This phase includes all hyperparameter tuning (using the validation window), final model selection, retraining the models, generating soft targets, and final evaluation on the ID and OOD test windows. **All final results reported in the thesis are derived from these full M5 experiments.**
- **Student Model Architecture**: A custom PyTorch/Lightning module using standard PyTorch **`nn.TransformerEncoder`** with a direct linear projection head to generate the 28-day forecasts.
- **Soft Target Generation**: **Offline pre-computation**. The trained TFT teacher's forecasts on the training and validation splits will be computed once, saved to disk, and loaded during student training to optimize speed and resource usage.
- **Loss Functions**: **Huber Loss (Smooth L1)** for both the supervised loss ($L_{supervised}$) and the distillation loss ($L_{distillation}$) to provide robustness against demand spikes and outliers.
- **Evaluation Framework**: Implement the **official M5 hierarchical WRMSSE framework** (adapted to the CA_1 subset hierarchy for Phase 1 code verification, and applied to all 12 levels of the full M5 hierarchy for the Phase 2 experiments). Additionally, track MAE, RMSE, MASE, WAPE, deployment efficiency metrics (trainable parameters, model size, inference speed, and training time), and relative ID-to-OOD degradation. Final evaluation metrics are reported strictly on the **ID Test** and **OOD Test** windows (the validation split is excluded from final reports).

