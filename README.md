# Audio processing utils

Audio processing utilities for traininig.
Implemented in one file for more convenient usability.

This repository will be greatly useful for tasks:
- Voice Conversion
- some others without utterences

## Installation

On some platforms:

**google colab/jupyter notebook:**
```python
!git clone https://github.com/SolomidHero/audio-utils.git audio-utils
!pip install -r audio-utils/requirements.txt

import sys
sys.path.append(0, "/content/audio-utils") # path to cloned repository
```

## Usage

### Python shell

Assume, you have data in **`./data/some_data_folder`**. Your steps now consist of:
```python
from utils import preprocess, PickleDataset

# create files in ./data/some_data_folder-preprocessed/
# and its description ./data/some_data_folder-preprocessed-info.csv
preprocess('./data/some_data_folder', pattern='.wav')

# create dataset with .csv file
dataset = PickleDataset('./data/some_data_folder-preprocessed-info.csv', pattern='.wav') # torch API Dataset

# tensor (n_mels, segment_size), tensor (n_mels, segment_size), tensor (n_mels, segment_size)
dataset[42]
```

Now with this dataset you can pass it wherever to obtain dataloader, subsets, etc.

### As preprocessing script

Preprocessing dataset:
```bash
python3 preprocess.py --data_root=./data/some_data_folder --pattern=.wav --n_jobs=4 --engine=torch
```

Slashing audios into piecies:
```bash
python3 slash.py --maxlen 1 --drop_last file1 file2.wav
# file1     -> file1_0s,     file1_1s, ...
# file2.wav -> file2_0s.wav, file2_1s.wav, ...
```