# README.md

## EEG Sleep Analysis Toolkit

This project provides tools to analyze, summarize, and visualize polysomnography (PSG) data from 20 patients. Data includes EEG, EOG, EMG, airflow, and temperature signals, annotated with sleep stages. The goal is to support practical research and exploration of sleep patterns and physiological signals.

### Features

#### Task 1: Data Structuring
- Unified data structure for all recordings
- 30s epoch labeling with sleep stages
- Signal normalization/standardization
- Multi-resolution access:
  - Epoch-level (30s)
  - Minute-level (aggregated epochs)
  - Sleep-stage-level (grouped by epochs)

#### Task 2: Sleep Metrics
- Query by patient, time range, or sleep stage
- Summary statistics:
  - Time in each sleep stage
  - Sleep efficiency
  - Mean and variance per physiological signal and stage

#### Task 3: Visualization Dashboard
- Interactive hypnograms (sleep stage vs. time)
- Time-series viewers for physiological data
- Summary plots:
  - Bar charts (time per stage)
  - Box plots (EMG signal variation across stages)
- Filters:
  - Patient
  - Sleep stage
  - Temporal resolution

### Requirements
- Python 3.8+
- Pandas, NumPy, Matplotlib, Plotly/Dash or Streamlit

### Dataset Access
Data is available upon request. Contact the project maintainers.

### How to Run
1. Load the dataset using provided scripts.
2. Use CLI or dashboard UI to explore and analyze data.
3. Generate visual summaries for research or presentation.


