I would like to perform a repository-wide audit before making any preprocessing changes. Please **do not modify any code yet**. Instead, investigate whether several high-memory columns are actually required after preprocessing.

### Background

A memory profile of the preprocessed DataFrame shows that the following columns consume a substantial proportion of memory:

* `date` (~3.5 GB)
* `d` (~3.2 GB)
* `wm_yr_wk` (~0.5 GB)

Together they account for a large fraction of the DataFrame memory. Since the model already uses `time_idx`, `weekday`, `month`, `year`, and other engineered features, I suspect these raw columns may no longer be required after preprocessing.

Before making any implementation changes, I want to verify this rigorously.

---

## Task 1 – Repository-wide column usage audit

Trace the usage of the following columns throughout the entire repository:

* `date`
* `d`
* `wm_yr_wk`

For each column, report:

1. Which file(s) reference the column.
2. Whether it is used only during preprocessing (e.g., merging or feature engineering).
3. Whether it is required by:

   * `build_timeseries_dataset`
   * `TimeSeriesDataSet`
   * training scripts
   * soft target generation
   * evaluation
   * inference
4. Whether it is only carried forward into the cached Parquet without any downstream use.

Please provide a summary table similar to:

| Column | Used During Preprocessing | Used After Preprocessing | Safe to Drop After Feature Engineering? | Reason |
| ------ | ------------------------- | ------------------------ | --------------------------------------- | ------ |

---

## Task 2 – Verify TimeSeriesDataSet requirements

Please inspect the implementation of our dataset construction pipeline and determine:

* Which DataFrame columns are actually consumed by `build_timeseries_dataset()`.
* Which columns are passed to `TimeSeriesDataSet`.
* Whether `date`, `d`, or `wm_yr_wk` are required by the dataset construction process.
* Whether the dataset builder depends only on `time_idx` and the configured feature lists.

Please reference the relevant code locations when explaining your findings.

---

## Task 3 – Quantify expected memory savings

Assuming each of the three columns is removed immediately after all required feature engineering is complete, estimate:

* Current DataFrame memory usage.
* Estimated memory after removing:

  * `date`
  * `d`
  * `wm_yr_wk`
* Percentage reduction.
* Whether this reduction is likely to significantly reduce the peak memory footprint during `TimeSeriesDataSet` construction.

The estimate does not need to be exact, but it should be technically justified.

---

## Task 4 – Recommendation

Based on the audit, recommend one of the following:

1. **No action** – the columns are required.
2. **Drop only `date`.**
3. **Drop `date` and `d`.**
4. **Drop `date`, `d`, and `wm_yr_wk`.**

If you recommend dropping any columns, specify the exact location in the preprocessing pipeline where they should be removed so that no downstream functionality is affected.

Please do not implement the changes yet. I want the audit findings first so we can decide whether this simpler optimization should be attempted before pursuing the larger per-store preprocessing refactor.
