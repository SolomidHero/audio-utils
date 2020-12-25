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
#    initial                 preprocess                training/inference time
#    data [wav] (on disk) -> data [specs] (on disk) -> augmented data [features] (in RAM)
#


import librosa
import librosa.display

import torch
import torchaudio
import torchaudio.transforms as tf

import matplotlib.pyplot as plt
import numpy as np

##############
# Parameters #
##############
sample_rate = 16000
stft_params = {
  'n_fft':      1024,
  'hop_length': 256,
  'win_length': 1024,
}
power = 1.0
amplitude_top_db = 80.0
n_mels = 80
mel_params = {
  'n_mels': n_mels,
  'fmax': 12000,
  'fmin': 0,
}

######################
# Data visualisation #
######################
def plot_spec(spec, title='Mel spectrogram', db=True, y_axis='linear'):
  fig = plt.figure(figsize=(14,5))
  if db:
    spec = librosa.power_to_db(spec, top_db=amplitude_top_db)
  librosa.display.specshow(spec, x_axis='time', y_axis=y_axis, cmap='viridis')

  plt.colorbar()
  plt.title(title)

  return fig

def plot_mel(*args, **kwargs):
  return plot_spec(*args, **kwargs)

def plot_mel_from_wav(path=None, data=None, sr=None, title='Mel spectrogram', db=True):
  mel = wav_to_mel(path, data, sr)
  plot_mel(mel, title=title, db=db)


########################
# Converting functions #
########################
def wav_to_mel(path=None, wav=None, sr=sample_rate, engine='librosa'):
  if path is None and wav is None:
    raise ValueError

  if path is not None:
    wav, sr = librosa.core.load(path)

  if engine == 'librosa':
    return librosa.feature.melspectrogram(wav, sr=sr, **stft_params, **mel_params, power=power)
  elif engine == 'torch':
    return tf.MelSpectrogram(sample_rate=sr, **stft_params, f_max=mel_params['fmax'], f_min=mel_params['fmin'], power=power)(
      torch.from_numpy(wav)
    )

  raise ValueError(engine)

def wav_to_spec(path=None, wav=None, sr=sample_rate, engine='librosa'):
  ''' STFT Spectrogram with absolute values '''
  if path is None and wav is None:
    raise ValueError

  if path is not None:
    wav, sr = librosa.core.load(path)

  if engine == 'librosa':
    return np.abs(librosa.stft(wav, **stft_params))
  elif engine == 'torch':
    return tf.Spectrogram(sample_rate=sr, **stft_params, power=power)(torch.from_numpy(wav))

  raise ValueError(engine)

def spec_to_wav(spec, sr=sample_rate, engine='librosa'):
  ''' using Griffin-Lim algorithm '''

  if engine == 'librosa':
    return librosa.griffinlim(spec, hop_length=stft_params['hop_length'], win_length=stft_params['win_length'])
  elif engine == 'torch':
    return tf.GriffinLim(**stft_params, power=power)(spec)

  raise ValueError(engine)

def mel_to_wav(mel, sr=sample_rate, engine='librosa'):
  ''' using Griffin-Lim algorithm '''

  if engine == 'librosa':
    return librosa.feature.inverse.mel_to_audio(mel, sr=sr, **stft_params)
  elif engine == 'torch':
    return spec_to_wav(tf.InverseMelScale(
      n_stft=stft_params['n_fft']//2 + 1, sample_rate=sample_rate, n_mels=n_mels,
      f_max=mel_params['fmax'], f_min=mel_params['fmin'], max_iter=1000
    )(mel), sr=sr, engine=engine)

  raise ValueError(engine)

def pitch_shift_wav(wav, shift=0, sr=sample_rate):
  return librosa.effects.pitch_shift(wav, sr, shift)


######################
# Dataset processing #
######################
# Functions and Classes that use torch API and disk memory to perform
# - preprocessing before training
# - additional processing (augmentations) during training/inference