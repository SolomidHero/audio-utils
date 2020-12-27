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

```
TODO: write fast installation guide
```

## How to use

Assume, you have data in **`./data/some_data_folder`**. Your steps now consist of:
```python
from utils import preprocess, PickleDataset

preprocess('./data/some_data_folder', pattern='.wav') # creates files in ./data/preprocessed/ with info.csv
dataset = PickleDataset('./data/preprocessed/info.csv', pattern='.wav') # torch API Dataset

# tensor (n_mels, segment_size), tensor (n_mels, segment_size)
dataset[42]
```

Now with this dataset you can pass it wherever to obtain dataloader, subsets, etc.