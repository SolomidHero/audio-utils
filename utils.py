# utils.py
# audio processing essential (simplified) functions
#
# base usage:
#   - get features from wav file                (preprocess)
#   - store them on disk                        (data pickles)
#   - load as dataset                           (pickles loading)
#   - process for training/inference/target     (process)
#   - get wav from processed data               (inverse)
#
# code namings:
#   wav  - raw data (array of floats) or path to .wav file (string)
#   spec - stft spectrogram, (2d array of floats)
#   mel  - mel spectrogram, (2d array of floats)
#   db   - db scale of mel or spec (10 * power * log_10(S / ref))
#
# engines:
#   librosa - operates with numpy arrays
#   torch   - operates with torch tensors (both cpu and gpu)
#
# how it works:
#   initial                 preprocess                             training/inference time
#   data [wav] (on disk) -> preprocessed data [specs] (on disk) -> load data/aug data [features] (in RAM)
#                           augmented data    [specs] (on disk)


import librosa
import librosa.display

import torch
import torchaudio

import torchaudio.transforms as tf
import audiomentations

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

##############
# Parameters #
##############
sample_rate = 16000
stft_params = {
  'n_fft':      2048,
  'hop_length': 256,
  'win_length': 1024,
}
power = 1.0
top_db_level = 80.0
n_mels = 80
mel_params = {
  'n_mels': n_mels,
  'fmax': sample_rate / 2,
  'fmin': 0,
}
segment_size = 200

######################
# Data visualisation #
######################
def plot_spec(spec, title='Mel spectrogram', db=True, y_axis='linear'):
  fig = plt.figure(figsize=(14,5))
  if db:
    spec = librosa.power_to_db(spec, top_db=top_db_level)
  librosa.display.specshow(spec, x_axis='time', y_axis=y_axis, cmap='viridis')

  plt.colorbar()
  plt.title(title)

  return fig

def plot_mel(*args, **kwargs):
  return plot_spec(*args, **kwargs)

def plot_mel_from_wav(path=None, data=None, sr=None, title='Mel spectrogram', db=True):
  mel = _wav_to_mel(path, data, sr)
  plot_mel(mel, title=title, db=db)


########################
# Converting functions #
########################
def _wav_to_mel(path=None, wav=None, sr=sample_rate, augment_fn=None, engine='librosa'):
  if path is None and wav is None:
    raise ValueError

  if path is not None:
    wav, _ = librosa.core.load(path, sr=None)

  if augment_fn is not None:
    wav = augment_fn(wav, sr)

  if engine == 'librosa':
    return librosa.feature.melspectrogram(wav, sr=sr, **stft_params, **mel_params, power=power)
  elif engine == 'torch':
    return tf.MelSpectrogram(sample_rate=sr, **stft_params, f_max=mel_params['fmax'], f_min=mel_params['fmin'], power=power)(
      torch.from_numpy(wav)
    )

  raise ValueError(engine)

def _wav_to_spec(path=None, wav=None, sr=sample_rate, engine='librosa'):
  ''' STFT Spectrogram with absolute values '''
  if path is None and wav is None:
    raise ValueError

  if path is not None:
    wav, _ = librosa.core.load(path, sr=None)

  if engine == 'librosa':
    return np.abs(librosa.stft(wav, **stft_params))
  elif engine == 'torch':
    return tf.Spectrogram(**stft_params, power=power)(torch.from_numpy(wav))

  raise ValueError(engine)

def _spec_to_wav(spec, sr=sample_rate, engine='librosa'):
  ''' using Griffin-Lim algorithm '''

  if engine == 'librosa':
    return librosa.griffinlim(spec, hop_length=stft_params['hop_length'], win_length=stft_params['win_length'])
  elif engine == 'torch':
    return tf.GriffinLim(**stft_params, power=power)(spec)

  raise ValueError(engine)

def _mel_to_wav(mel, sr=sample_rate, engine='librosa'):
  ''' using Griffin-Lim algorithm '''

  if engine == 'librosa':
    return librosa.feature.inverse.mel_to_audio(mel, sr=sr, **stft_params, power=power)
  elif engine == 'torch':
    return _spec_to_wav(tf.InverseMelScale(
      n_stft=stft_params['n_fft']//2 + 1, sample_rate=sr, n_mels=n_mels,
      f_max=mel_params['fmax'], f_min=mel_params['fmin'], max_iter=1000
    )(mel), sr=sr, engine=engine)

  raise ValueError(engine)

def _to_db(spec, engine='librosa'):
  if engine == 'librosa':
    return librosa.power_to_db(spec, top_db=top_db_level)
  elif engine == 'torch':
    return tf.AmplitudeToDB('power', top_db=top_db_level)(spec)

def _from_db(db_spec, engine='librosa'):
  if engine == 'librosa':
    return librosa.db_to_power(db_spec)
  elif engine == 'torch':
    return torch.pow(10.0, db_spec / 10)


#################
# Augmentations #
#################
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

# apply to:   wav
# when:       preprocessing
# results:    augmented data
_soft_augment_fn = Compose([
  AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
  PitchShift(min_semitones=-12, max_semitones=12, p=0.5),
])



######################
# Data preprocessing #
######################
# Purposes:
# - preprocess data
# - augment data
# - store on disk preprocessed and augmented data

def _find_files(directory, pattern='.wav', use_dir_name=True):
  '''
  Find files recursively
  Args:
    directory (str): root directory to find
    pattern (str): query to find
    use_dir_name (bool): if False, directory name is not included
  Return:
    (list): list of found filenames
  '''
  files = []
  for root, dirnames, filenames in os.walk(directory, followlinks=True):
    for filename in filenames:
      if filename[-len(pattern):] == pattern:
        files.append(os.path.join(root, filename))
  if not use_dir_name:
    files = [f.replace(directory + '/', '') for f in files]
  return files


def _preprocess(data_root, output_root=None, augment_fn=_soft_augment_fn, pattern='.wav', engine='librosa'):
  '''
  Preprocess each element in dataset with _wav_to_mel
  and store them on disk with appropriate name
  '''

  filenames = _find_files(data_root, pattern=pattern)
  output_pathes = []
  features_lengths = []

  if output_root == None:
    output_root = os.path.join(os.path.abspath(os.path.dirname(data_root)), "preprocessed")

  if not os.path.exists(output_root):
    os.mkdir(output_root)

  for fname in tqdm(filenames):
    output_path = os.path.join(output_root, os.path.basename(fname).replace(pattern, ''))

    # preprocess
    feature_len = _extract_features(fname, output_path, engine=engine)

    # augment
    if augment_fn is not None:
      _extract_features(fname, output_path + '-augmented', augment_fn=augment_fn, engine=engine)

    if feature_len < segment_size:
      continue
    output_pathes.append(output_path)
    features_lengths.append(feature_len)

  # save information about data
  pd.DataFrame(data={
    'path': list(map(lambda x: x + '.npy', output_pathes)),
    'augmented_path': None if augment_fn is None else list(map(lambda x: x + '-augmented.npy', output_pathes)),
    'size': sorted(features_lengths),
    'label': 'None'
  }).sort_values(by=['size']).to_csv(os.path.join(output_root, 'info.csv'))


def _extract_features(file_path, output_path=None, augment_fn=None, delta=False, engine='librosa'):
  '''
  Preprocess each element in dataset with _wav_to_mel
  and store them on disk with appropriate name
  '''

  if engine == 'librosa':
    mel = _to_db(_wav_to_mel(file_path, augment_fn=augment_fn, engine=engine), engine=engine)
  elif engine == 'torch':
    mel = _to_db(_wav_to_mel(file_path, augment_fn=augment_fn, engine=engine), engine=engine).numpy()


  features = [mel]
  if delta:
    features.append(librosa.feature.delta(features[0]))
  features = np.concatenate(features, axis=0).astype(np.float32)

  if output_path is not None:
    np.save(output_path, features)
    return features.shape[1] # time dimension size
  else:
    return features


###########################
# Dataset and Dataloading #
###########################
# Classes that use torch API and disk memory to effectively perform
# - loading the data into RAM
# - slashing all-time input on fixed time segments
# - additional processing (augmentations) during training/inference
class PickleDataset(torch.utils.data.Dataset):
  def __init__(self, csv_path="./preprocessed/info.csv", segment_size=segment_size, return_type='torch'):
    super().__init__()
    self.info = pd.read_csv(csv_path)
    self.segment_size = segment_size
    self.return_type = return_type

  def __getitem__(self, i):
    '''
    Returns:
      - segment of preprocessed source (tensor shape (n_mels, segment_size))
      - segment of augmented source (tensor shape (n_mels, segment_size))
    '''

    src = np.load(self.info['path'].iloc[i])
    t, t_aug = np.random.randint(0, src.shape[-1] - self.segment_size, size=2)

    result = [src[:, t:t + self.segment_size]]
    if self.info['augmented_path'].iloc[i] is not None:
      aug_src = np.load(self.info['augmented_path'].iloc[i])
      result.append(aug_src[:, t_aug:t_aug + self.segment_size])

    if self.return_type == 'torch':
      result = map(torch.from_numpy, result)

    return tuple(result)

  def __len__(self):
    return len(self.info)

