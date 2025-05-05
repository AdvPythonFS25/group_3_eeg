
import os
import typing
import mne
import dataclasses
import numpy as np
from enum import Enum, auto
from collections import Counter

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
          
        print(f"----------------- {epochs}")

    return results
    


  def generate_summary_stats(self, filter):
    '''
    Task 2
    Per-patient:
     time spent in each stage,
     sleep egiciency,
     mean,
     variance of each physiological channel per stage,

     i imagine this function would look like
      ```
     filtered = self.query(filter)
     for sample in filtered: sample.plot_summary(filter)
     ```
    '''
  
    """
    for sample in filtered:
      id_to_label = {v: k for k, v in sample.event_id.items()}
      stage_ids = sample.events[:, 2]  # grab all stage IDs in the sample
      label_counts = Counter(id_to_label[stage_id] for stage_id in stage_ids)
    
      for label, count in label_counts.items():
        stage_counts[label] = stage_counts.get(label, 0) + count
      
      time_spent = {k : f"{v * 30} seconds" for k, v in stage_counts.items() }"""
    
    stage_counts = {}
      
    for file_id in self.index.keys():
      print(file_id)
      sample = self.samples[file_id]
      id_to_label = {v: k for k, v in sample.epo.event_id.items()}
      stage_ids = sample.epo.events[:, 2]  # grab all stage IDs in the sample
      label_counts = Counter(id_to_label[stage_id] for stage_id in stage_ids)
    
      for label, count in label_counts.items():
        stage_counts[label] = stage_counts.get(label, 0) + count
      
      time_spent = {k : f"{v * 30} seconds" for k, v in stage_counts.items() }
      
        
    return time_spent

class AccessType(Enum):
  '''Type to set temporal resolution for accessing EEG sample'''
  Epoch = auto()
  Minute = auto()

class EEG_Sample:
  def __init__(self, path, access_pattern=AccessType.Epoch):
    self.access_pattern = access_pattern
    self.epo = mne.read_epochs(path, preload=False)
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
    data = self.data()
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
  """print(dataset.query({
    'age': 28,
    'sex': 'Male',
    'time_range': (600, 1800),
    'sleep_stages': [2, 0]
  }))"""
  #print(dataset.samples.items())
  print(dataset.generate_summary_stats({
    'age': 28,
    'sex': 'Male',
    'time_range': (600, 2000),
    'sleep_stages': [1,2, 0]
  }))

if __name__ == '__main__':
  main()