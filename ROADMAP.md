# ROADMAP.md

## EEG Sleep Analysis Project Roadmap

### Phase 1 – Data Preparation
- [x] Request and load dataset
- [x] Create unified data structure for all signals
- [x] Label 30s epochs with sleep stages
- [x] Normalize/standardize signals
- [x] Implement temporal resolution access (epoch, minute, stage)

### Phase 2 – Metrics Engine
- [x] Build querying interface for patient/time/stage filters
- [x] Compute time spent in each sleep stage
- [x] Calculate sleep efficiency per patient
- [x] Aggregate mean/variance of signals by stage

### Phase 3 – Dashboard Development
- [x] Create hypnogram visualizer with patient selector
- [ ] Build interactive time-series plots with range selectors
- [ ] Add bar/box plot generators for summary stats
- [ ] Implement filters (patient, stage, resolution)
