
import os
import typing
import mne
import dataclasses
import numpy as np
from enum import Enum, auto
from collections import Counter, defaultdict
from matplotlib import pyplot as plt

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

        info = record['info']
        print(f"-----------------------{info}")
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

        #Manipulation of mne objects starts here 
        sfreq = info['sfreq']
        
        if 'time_range' in filters:
          start, end = filters['time_range']

           # Convert absolute sample positions to seconds
          epoch_times = epochs.epo.events[:, 0] / sfreq

          # Normalize to relative time (start from 0) 
          relative_times = epoch_times - epoch_times[0]

          #Mask for epochs within the desired relative time range
          time_mask = (relative_times >= start) & (relative_times <= end)
          selected_indices = np.where(time_mask)[0]

          if len(selected_indices) > 0: 
            epochs = epochs[selected_indices]
        
        #keep x axis lenght of time sliced object
        original_labels = epochs.epo.events[:, 2]
        masked_labels = None
        if 'sleep_stages' in filters:
          desired_stages = filters['sleep_stages']  # e.g., [1, 2]
          masked_labels = np.array(original_labels, dtype=float)  # cast to float so NaNs work
          masked_labels[~np.isin(masked_labels, desired_stages)] = np.nan

        if masked_labels is not None and masked_labels.any():
          # safe to use masked_labels
          results.append((epochs, masked_labels))
        else:
          results.append((epochs, None))
          
      # single of multiple sample selection after query from user input
        
    print("Multiple matching samples found:")
    for i, sample in enumerate(results):
        print(f"{i}: {sample[0]}")

    while True:
        try:
            selection = int(input(f"Select a sample (0–{len(results)-1}): "))
            if 0 <= selection < len(results):
                selected_sample = results[selection]
                break
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid integer.")

    return selected_sample
    

  def generate_summary_stats(self):
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
      data = sample.data()

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
  def __init__(self, epo_or_path, access_pattern=AccessType.Epoch):
    self.access_pattern = access_pattern
    if isinstance(epo_or_path, mne.BaseEpochs): 
        self.epo = epo_or_path
    elif isinstance(epo_or_path, str):
        self.epo = mne.read_epochs(epo_or_path, preload=False)
    else:
        raise TypeError(f"EEG_Sample must be initialized with a path or an mne.Epochs object, got {type(epo_or_path)}")

  def __getitem__(self, idx):
    # Slice the underlying epochs and return a new EEG_Sample
    return EEG_Sample(self.epo[idx])

  def data(self):
    return self.epo.get_data()
    
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

def hypnogram(dataset):
  ## Get user inputs for query function

  user_input = input("Enter age (e.g., 25), or None: ")

  if user_input.lower() == 'none':
    age = None
  else:
    age = int(user_input)
    if age <= 0:
        age = None
    
    # Ask for sex
  sex = input("Enter sex (e.g., Male/Female) or None: ").strip()
  if sex == "None":
    sex = None
    
    # Ask for time range
  time_range_input = input("Enter time range in seconds as tuple min 0, max 28890 e.g. (40, 60) or None: ").strip()

  if time_range_input.lower() == "none":
    time_range = None
  else:
    try:
      # Safely evaluate the tuple input
      time_range = eval(time_range_input, {"__builtins__": {}}, {})
      if (isinstance(time_range, tuple) and 
          len(time_range) == 2 and 
          all(isinstance(x, int) for x in time_range) and
          0 <= time_range[0] < time_range[1] <= 1000000000):
            pass  # valid
      else:
         raise ValueError("Invalid time range")
    except Exception as e:
        print(f"Invalid input: {e}")
        time_range = None
    
    # Ask for sleep stages
  sleep_stages = input("Enter sleep stages (comma-separated, e.g.,  0, 1, 2, 3, 4, where 0 is awake and 4 is REM or None: ").strip()
  if sleep_stages == 'None':
    sleep_stages = None
  else:
    sleep_stages = [int(stage.strip()) for stage in sleep_stages.split(',')]
  
  query_params = {}
  if age is not None:
    query_params['age'] = age
  if sex is not None:
    query_params['sex'] = sex
  if time_range is not None:
    query_params['time_range'] = time_range
  if sleep_stages is not None:
    query_params['sleep_stages'] = sleep_stages
    
  query_result = dataset.query(query_params)
  
  selected_epochs, labels = query_result
  if labels is None:
    labels = selected_epochs.epo.events[:, 2]
  x = np.arange(len(labels))
  stage_labels = ['W', '1', '2', '3/4', 'R']
  plt.figure(figsize=(12, 3))
  plt.plot( x, labels, drawstyle='steps-post')
  plt.yticks(ticks=[0, 1, 2, 3, 4], labels=stage_labels)
  plt.xlabel('Epoch Index (30s intervals)')
  plt.ylabel('Sleep Stage')
  plt.title('Hypnogram (Sleep Stages Over Time)')
  plt.grid(True)
  plt.tight_layout()
  plt.show()
  
def stats_visualizers(dataset):
  test = dataset.generate_summary_stats()
  
  stage_time_totals = defaultdict(list)

  for stats in test.values():
    for stage, time in stats["time_spent"].items():
        stage_time_totals[stage].append(time)

  # Compute average time per stage
  avg_stage_times = {stage: np.mean(times) for stage, times in stage_time_totals.items()}

  # Plot
  stages = list(avg_stage_times.keys())
  avg_times = [avg_stage_times[stage] for stage in stages]

  plt.figure(figsize=(10, 6))
  plt.bar(stages, avg_times, color='skyblue')
  plt.xlabel("Sleep Stage")
  plt.ylabel("Average Time Spent (s)")
  plt.title("Average Time Spent in Each Sleep Stage")
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()
  

  emg_variances_by_stage = defaultdict(list)
  emg_channel = "EMG submental"

  for stats in test.values():
    signal_stats = stats["signal_stats"]
    for stage, channels in signal_stats.items():
        if emg_channel in channels:
            emg_variances_by_stage[stage].append(channels[emg_channel]["variance"])

  # Prepare data for box plot
  stages = list(emg_variances_by_stage.keys())
  data = [emg_variances_by_stage[stage] for stage in stages]

  plt.figure(figsize=(10, 6))
  plt.boxplot(data, labels=stages, showfliers=False)
  plt.ylabel("EMG Signal Variance")
  plt.title("EMG Signal Variance Across Sleep Stages")
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

def main():
  dataset = EEG_Dataset('./data')

  # compare access patterns
  sample = dataset.samples['PHY_ID0000-epo.fif']
  sample.set_access_pattern(AccessType.Epoch)
  t = sample[0:10]
  sample.set_access_pattern(AccessType.Minute)
  t2 = sample[0:5]
  #assert np.equal(t, t2).all(), 'Access patterns do not match'
  
  
  hypnogram(dataset)
  stats_visualizers(dataset)
  #print(dataset.samples.items())
  #summary_stats_ex = dataset.generate_summary_stats()
  #print("For phase 2, we implemented our tasks in two methods as part of a greater class. Query patient allows you to query by a combination of none or all of the following attributes: age, sex, sleep stages, and time range. Summary statisitcs calculates time in each sleep stage, sleep efficiency, as well as the mean and variance per signal. Additionally, it calculates the mean and variance of each signal per sleep stage ")
  #print(f"Example of querying for an specific object using patient age and sex, as well as speciic slices of the queried object specifying sleep stage and times {query_ex}")
  #print(f"Example of summary stats for one mne object (from EEG_Dataset.generate_summary_stats): {summary_stats_ex['PHY_ID0005-epo.fif']}")

if __name__ == '__main__':
  main()