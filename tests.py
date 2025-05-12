import unittest
from eeg_analyzer import EEG_Dataset, AccessType, EEG_Sample
import numpy as np

class TestEEGSample__getitem__(unittest.TestCase):
  def setUp(self):
    dataset = EEG_Dataset('./test_data')
    self.sample = dataset.samples['random_test_data-epo.fif']

  def test_compare_access_patterns(self):
    sample = self.sample
    sample.set_access_pattern(AccessType.Epoch)
    t = sample[0:10]
    sample.set_access_pattern(AccessType.Minute)
    t2 = sample[0:5]
    self.assertIsInstance(t, EEG_Sample)
    self.assertIsInstance(t2, EEG_Sample)
    self.assertTrue(np.allclose(t.data(), t2.data()), 'Access patterns do not match')
  
  def test_epoch_access_slice(self):
        self.sample.set_access_pattern(AccessType.Epoch)
        result = self.sample[0:5]
        self.assertIsInstance(result, EEG_Sample)
        self.assertEqual(result.data().shape[0], 5)

  def test_minute_access_array(self):
    self.sample.set_access_pattern(AccessType.Minute)
    idx = np.array([0,1,1])
    result = self.sample[idx]
    self.assertIsInstance(result, EEG_Sample)
    self.assertEqual(result.data().shape[0], 6)  # Each index is repeated twice (masking with binary array should mask two epochs for each minute)

  def test_invalid_access_pattern(self):
    self.sample.set_access_pattern(AccessType.Minute)
    with self.assertRaises(ValueError):
      _ = self.sample["invalid"]

if __name__ == '__main__':
    unittest.main()