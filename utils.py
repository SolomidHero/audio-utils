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
#   db   - db scale of mel or spec (10 * log_10(S / ref))
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
from joblib import Parallel, delayed

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
n_jobs = 4

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
    return tf.MelSpectrogram(
      sample_rate=sr, **stft_params, n_mels=n_mels,
      f_max=mel_params['fmax'], f_min=mel_params['fmin'], power=power
    )(torch.from_numpy(wav))

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


###################
# Transformations #
###################
# Here defined transformations of source wav
# to perform preprocessing and augmentations
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

# apply to:   wav
# when:       augmentation in preprocessing stage
# results:    wav
_soft_augment_fn = Compose([
  AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
  PitchShift(min_semitones=-6, max_semitones=6, p=0.5),
])

# apply to:   wav path
# when:       preprocessing
# results:    mel
def _feature_extractor_fn(file_path, augment_fn=None, engine='librosa'):
  return _to_db(_wav_to_mel(file_path, augment_fn=augment_fn, engine=engine), engine=engine)


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


def _slash_wav(wav, sr, maxlen=10, drop_last=False, start_pos=0., n_parts=-1, return_timings=False):
  '''
  Break wav into segments

  Args:
    wav, sr         - wav array (first dim is time) and its sampling rate
    maxlen          - maximum length of segment in seconds
    drop_last       - if last incomplete segment should be dropped
    start_pos       - time (in seconds) from where to start slashing
    n_parts         - maximum total number of segments (should be more than 0)
    return_timings  - return corresponding start_pos of each segments
  Returns:
    list of wavs    - segments
  '''

  ticks = np.arange(0, len(wav) - sr * start_pos, sr * maxlen, dtype=int)
  times = np.arange(start_pos, len(wav) / sr, maxlen, dtype=int)
  segments = np.split(wav[sr * start_pos:], ticks[1:])

  if len(segments[-1]) == 0 or (drop_last and len(segments[-1]) < int(sr * maxlen)):
    segments = segments[:-1]

  if n_parts > 0 and isinstance(n_parts, int):
    segments = segments[:n_parts]

  if return_timings:
    return segments, times[:len(segments)]
  return segments


def preprocess(data_root, output_root=None, augment_fn=_soft_augment_fn, pattern='.wav', parallel=False, n_jobs=4, engine='librosa'):
  '''
  Preprocess each element in dataset with _wav_to_mel
  and store them on disk with appropriate name
  '''

  filenames = _find_files(data_root, pattern=pattern)
  output_pathes = []
  features_lengths = []

  if output_root == None:
    output_root = os.path.abspath(data_root) + "-preprocessed"

  if not os.path.exists(output_root):
    os.mkdir(output_root)

  output_pathes = list(map(
    lambda p: os.path.join(output_root, os.path.basename(p).replace(pattern, '')),
    filenames
  ))

  if not parallel:
    # preprocess
    features_lengths = [
      _extract_features(name, op, engine=engine)
      for name, op in tqdm(list(zip(filenames, output_pathes)))
    ]

    # augment
    if augment_fn is not None:
      for name, op in tqdm(list(zip(filenames, output_pathes))):
        _extract_features(name, op + '-augmented', augment_fn=augment_fn, engine=engine)

  else:
    # preprocess
    features_lengths = Parallel(n_jobs=n_jobs)(
      delayed(_extract_features)(name, op, engine=engine)
      for name, op in tqdm(list(zip(filenames, output_pathes)))
    )

    # augment
    if augment_fn is not None:
      Parallel(n_jobs=n_jobs)(
        delayed(_extract_features)(name, op + '-augmented', augment_fn=augment_fn, engine=engine)
        for name, op in tqdm(list(zip(filenames, output_pathes)))
      )

  output_pathes = [op for op, fl in zip(output_pathes, features_lengths) if fl >= segment_size]
  features_lengths = [fl for fl in features_lengths if fl >= segment_size]

  # save information about data
  pd.DataFrame(data={
    'path': list(map(lambda x: x + '.npy', output_pathes)),
    'augmented_path': None if augment_fn is None else list(map(lambda x: x + '-augmented.npy', output_pathes)),
    'size': sorted(features_lengths),
    'label': 'None'
  }).sort_values(by=['size']).to_csv(os.path.abspath(output_root) + "-info.csv")


def _extract_features(file_path, output_path=None, augment_fn=None, delta=False, engine='librosa'):
  '''
  Preprocess each element in dataset with _wav_to_mel
  and store them on disk with appropriate name
  '''

  mel = _feature_extractor_fn(file_path, augment_fn=augment_fn, engine=engine)
  if engine == 'torch':
    mel = mel.numpy()

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
  def __init__(self, csv_path, segment_size=segment_size, return_type='torch'):
    super().__init__()
    self.info = pd.read_csv(csv_path)
    self.segment_size = segment_size
    self.return_type = return_type

  def __getitem__(self, i):
    '''
    Returns:
      - instance  segment of preprocessed source (tensor shape (n_mels, segment_size))
      - reference segment of augmented source (tensor shape (n_mels, segment_size))
      - instance  segment of augmented source (tensor shape (n_mels, segment_size))
    '''

    src = np.load(self.info['path'].iloc[i])
    t, t_aug = np.random.randint(0, src.shape[-1] - self.segment_size + 1, size=2)

    result = [src[:, t:t + self.segment_size]]
    if self.info['augmented_path'].iloc[i] is not None:
      aug_src = np.load(self.info['augmented_path'].iloc[i])
      result.append(aug_src[:, t_aug:t_aug + self.segment_size])

    result.append(aug_src[:, t:t + self.segment_size])

    if self.return_type == 'torch':
      result = map(torch.from_numpy, result)

    return tuple(result)

  def __len__(self):
    return len(self.info)

