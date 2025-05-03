
import os
import typing
import mne
import dataclasses
from enum import Enum, auto

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

  def query(self, filter):
    '''Task 2
    Filter must support:
    - selection of patients
    - time ranges
    - sleep stages
    '''
    raise NotImplementedError()

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
    raise NotImplementedError()

class AccessType(Enum):
  '''Type to set temporal resolution for accessing EEG sample'''
  Epoch = auto()
  Minute = auto()

class EEG_Sample:
  def __init__(self, path, access_pattern=AccessType.Epoch):
    self.access_pattern = access_pattern
    self.epo = mne.read_epochs(path, preload=False)
    # ... 

  def data(self):
    # access by function to lazy load data
    if self.data is None:
      self.data = self.epo.get_data(copy=False)
    return self.data
    
  def set_access_pattern(self, new:AccessType):
    self.access_pattern = new

  def get_metadata(self):
    return self.epo['info']

  def summary(self):
    '''
    for task 2
    just plot all the relevant things here i think
    '''
    raise NotImplementedError()
  
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
    print(val)
    if self.access_pattern == AccessType.Epoch:
      return self.data[val]
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
  print(dataset.index, dataset.samples)

if __name__ == '__main__':
  main()