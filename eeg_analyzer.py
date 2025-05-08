
import os
import typing
import mne
import dataclasses
import numpy as np
from enum import Enum, auto
from collections import Counter, defaultdict

class EEG_Dataset:
  def __init__(self, path_to_data_directory):
    self.data_dir = path_to_data_directory
    self.generate_index()
  
  def generate_hypnograms(self, patient):
    '''Task 3'''
    return self.samples[patient].generate_hypnogram()

  def generate_index(self):
    '''Find all files; read and save their metadata.'''
    index = {}
    samples = {}
    for root, _, files in os.walk(self.data_dir):
      for file_path in filter(lambda f:f.endswith('.fif'), files):
        path_full = os.path.join(root, file_path)
        sample = EEG_Sample(path_full) 
        # generating some kind of basic info about each file
        # feel free to add more stuff here
        index[file_path] = {'info': sample.get_metadata()}
        samples[file_path] = sample

    self.index = index
    self.samples = samples

  def query(self, filters):
    '''Task 2
    Filter must support:
    - selection of patients
    - time ranges
    - sleep stages
    '''
    results = []

    for file_id, record in self.index.items():
        #print('-----------')
        #print(file_id)
        info = record['info']
        epochs = self.samples[file_id]  # MNE Epochs object

        # Extract subject metadata
        subject = info.get('subject_info', {})
        try:
            age = int(subject.get('last_name', '').replace('yr', ''))
        except:
            age = None
        sex = subject.get('first_name', '').lower()

        # patient filter
        if 'age' in filters and filters['age'] != age:
            continue
        if 'sex' in filters and filters['sex'].lower() != sex:
            continue
          
        #print(f"subject: {subject}")
        #print(self.samples[file_id])

        #print(f"--------------{epochs.epo.tmax}")
        #Sleep stage filtering
        if 'sleep_stages' in filters:
            #print(f"3333333333333{filters['sleep_stages']}")
            stage_mask = np.isin(epochs.epo.events[:, 2], filters['sleep_stages'])
            epochs = epochs[stage_mask]

        #print(f"2222222222222{epochs}")
        # Time range filtering 
        
        sfreq = info['sfreq']
        tmin = epochs.tmin  # typically 0
        times = epochs.events[:, 0] / sfreq
        #print(f"gggggggggg {times}")
        
        if 'time_range' in filters:
          start, end = filters['time_range']
          sfreq = epochs.info['sfreq']

           # Convert absolute sample positions to seconds
          epoch_times = epochs.events[:, 0] / sfreq

          # Normalize to relative time (start from 0) 
          relative_times = epoch_times - epoch_times[0]

          # Mask for epochs within the desired relative time range
          time_mask = (relative_times >= start) & (relative_times <= end)
          selected_indices = np.where(time_mask)[0]

        if len(selected_indices) > 0: 
          epochs = epochs[selected_indices]

        if epochs:
          results.append( epochs)

    return results
    

  def generate_summary_stats(self, filter):
    #generate summary statistics indicated in roadmap
    
    all_stage_counts = {}  # dictionary keyed by file_id

    for file_id in self.index.keys():
      sample = self.samples[file_id]
    
      id_to_label = {v: k for k, v in sample.epo.event_id.items()}
      stage_ids = sample.epo.events[:, 2]  # stage ID per epoch
      label_counts = Counter(id_to_label[stage_id] for stage_id in stage_ids)

      #  Time spent per stage in seconds (assuming 30s epochs)
      time_spent = {label: count * 30 for label, count in label_counts.items()}

      # Compute sleep efficiency
      wake_time = time_spent.get('Sleep stage W', 0)
      total_time = sum(time_spent.values())
      sleep_time = total_time - wake_time

      if sleep_time > 0:
          sleep_efficiency = sleep_time / total_time
      else:
        sleep_efficiency = None
        
      #Next loop is to calculate mean and variance of eahc object
      signal_stats = defaultdict(dict)      
      data = sample.data

      for stage_id in np.unique(stage_ids):
            stage_label = id_to_label[stage_id]
            # Indices of epochs for this stage
            stage_indices = np.where(stage_ids == stage_id)[0]
            stage_data = data[stage_indices]  # shape: (n_epochs_stage, n_channels, n_times)

            # Compute mean and variance per channel
            # Shape: (n_channels,)
            stage_mean = stage_data.mean(axis=(0, 2))  # mean over epochs and time
            stage_var = stage_data.var(axis=(0, 2))    # variance over epochs and time

            for ch_name, mean_val, var_val in zip(sample.epo.ch_names, stage_mean, stage_var):
                signal_stats[stage_label][ch_name] = {
                    'mean': mean_val,
                    'variance': var_val
                }  

      all_stage_counts[file_id] = {
        "stage_counts": dict(label_counts),
        "time_spent": time_spent,
        "sleep_efficiency": sleep_efficiency,
        "signal_stats": signal_stats
      }
        
    return all_stage_counts

class AccessType(Enum):
  '''Type to set temporal resolution for accessing EEG sample'''
  Epoch = auto()
  Minute = auto()

class EEG_Sample:
  def __init__(self, path, access_pattern=AccessType.Epoch):
    self.access_pattern = access_pattern
    self.epo = mne.read_epochs(path, preload=False)
    self.data = self.epo.get_data()
    # ... 

  """def data(self):
    # access by function to lazy load data
    if self.data is None:
      self.data = self.epo.get_data(copy=False)
    return self.data"""
    
  def set_access_pattern(self, new:AccessType):
    self.access_pattern = new

  def get_metadata(self):
    return self.epo.info

  def summary(self):
    '''
    for task 2
    just plot all the relevant things here i think
    '''
    
    
    
    return self.epo.event_id.items()
  
  def generate_hypnogram(self):
    raise NotImplementedError()

  
  def normalize(self):
    '''
    From task 1
    '''
    # just copy-pasted code from notebook here. Not tested.
    data = self.data
    mean = data.mean(axis=2, keepdims=True)
    std = data.std(axis=2, keepdims=True)
    normalized = (data - mean) / std
    self.normalized_data = normalized
  
  def __getitem__(self, val):
    """
    Allows us to do stuff like `eeg_sample[10:20]`
    Also this + AccessType takes care of 1) and 2) of the first task
    TODO: handle the minute-level by aggregating two epochs
    """
    #print(val)
    if self.access_pattern == AccessType.Epoch:
      return self.epo[val]
    if self.access_pattern == AccessType.Minute:
      raise NotImplementedError()

  def group_per_sleep_stage(self):
    '''
    For last part of the first task
    '''
    raise NotImplementedError()

class SleepStage(Enum):
  W = auto()
  s_1 = auto()
  s_2 = auto()
  s_34 = auto()
  R = auto()
  

import typing as t
@dataclasses.dataclass
class Filter:
  patient:t.Set[str]
  access_type:AccessType
  sleepStage:t.Set[SleepStage]
  def apply(self, dataset ):
    '''
    TODO: apply filter to dataset here
    '''

def main():
  dataset = EEG_Dataset('./data')
  
  #print(dataset.index.items())
  query_ex = dataset.query({
    'age': 28,
    'sex': 'Male',
    'time_range': (600, 1800),
    'sleep_stages': [2, 0]
  })
  #print(dataset.samples.items())
  summary_stats_ex = dataset.generate_summary_stats({
    'age': 28,
    'sex': 'Male',
    'time_range': (600, 2000),
    'sleep_stages': [1,2, 0]
  })
  print("For phase 2, we implemented our tasks in two methods as part of a greater class. Query patient allows you to query by a combination of none or all of the following attributes: age, sex, sleep stages, and time range. Summary statisitcs calculates time in each sleep stage, sleep efficiency, as well as the mean and variance per signal. Additionally, it calculates the mean and variance of each signal per sleep stage ")
  print(f"Example of querying for an specific object using patient age and sex, as well as speciic slices of the queried object specifying sleep stage and times {query_ex}")
  print(f"Example of summary stats for one mne object (from EEG_Dataset.generate_summary_stats): {summary_stats_ex['PHY_ID0005-epo.fif']}")

if __name__ == '__main__':
  main()