# utils.py
# preprocess function from utils.py as standalone script
#

import argparse
from utils import preprocess, _soft_augment_fn

def get_args():
  parser = argparse.ArgumentParser(description='preprocess dataset folder (with augment defined in utils.py)')

  parser.add_argument('--data_root', type=str, help='Path to raw dataset')
  parser.add_argument('--output_root', default=None, type=str, help='Path to store preprocessed data', required=False)

  parser.add_argument('--engine', default='librosa', type=str, help="Preprocessing library ('librosa' or 'torch')", required=False)
  parser.add_argument('--pattern', default='.wav', type=str, help='Extension of data files', required=False)
  parser.add_argument('--delta', default=True, type=bool, help='Append delta of spectrogram feature', required=False)

  parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)

  args = parser.parse_args()
  return args


def main():
  # get arguments
  args = get_args()

  preprocess(
    args.data_root,
    output_root=args.output_root,
    augment_fn=_soft_augment_fn,
    pattern=args.pattern,
    parallel=True,
    n_jobs=args.n_jobs,
    engine=args.engine
  )

if __name__ == '__main__':
  main()