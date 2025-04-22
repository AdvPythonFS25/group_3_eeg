# ROADMAP.md

## EEG Sleep Analysis Project Roadmap

### Phase 1 – Data Preparation
- [x] Request and load dataset
- [x] Create unified data structure for all signals
- [x] Label 30s epochs with sleep stages
- [x] Normalize/standardize signals
- [ ] Implement temporal resolution access (epoch, minute, stage)

### Phase 2 – Metrics Engine
- [ ] Build querying interface for patient/time/stage filters
- [ ] Compute time spent in each sleep stage
- [ ] Calculate sleep efficiency per patient
- [ ] Aggregate mean/variance of signals by stage

### Phase 3 – Dashboard Development
- [ ] Create hypnogram visualizer with patient selector
- [ ] Build interactive time-series plots with range selectors
- [ ] Add bar/box plot generators for summary stats
- [ ] Implement filters (patient, stage, resolution)
