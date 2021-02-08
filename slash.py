# slash.py
# _slash_wav function from utils.py as a standalone script
# apply:      to a single file
# creates:    filename_0s.wav, filename_10s.wav, ...


import argparse
import os
import librosa
import scipy
from utils import _slash_wav

def get_args():
  parser = argparse.ArgumentParser(description='break a single audio file into segments')

  parser.add_argument('file', metavar='filepath', type=str, nargs='+', help='Path to audio file')
  parser.add_argument('--output_root', default=None, type=str,
    help='Path to store slashed audios (default: same dirictory)', required=False
  )

  parser.add_argument('--maxlen', default=10., type=float, help='Maximum length of segment (in seconds)', required=False)
  parser.add_argument('--drop_last', action='store_true', help='Whether to drop last incomplete segment', required=False)
  parser.add_argument('--rate', default=None, type=int, help='Resample audio before slashing', required=False)

  parser.add_argument('-s', '--start', default=0., type=float, help='Start position for processing (in seconds)', required=False)
  parser.add_argument('-N', '--n_parts', default=-1, type=int, help='Total number of resulting segments (could be less)', required=False)

  args = parser.parse_args()
  return args


def main():
  # get arguments
  args = get_args()

  for file_path in args.file:
    dirname, filename = os.path.split(file_path)
    filename, file_extension = os.path.splitext(filename)
    dirname = dirname if args.output_root is None else args.output_root

    y, sr = librosa.core.load(file_path, sr=None)

    # resample if specified
    if args.rate is not None:
      y = librosa.core.resample(y, sr, args.rate)
      sr = args.rate

    # slash wav into segments
    wavs, timings = _slash_wav(
      y, sr,
      maxlen=args.maxlen,
      drop_last=args.drop_last,
      start_pos=args.start,
      n_parts=args.n_parts,
      return_timings=True,
    )

    # output into corresponding files
    for timing, wav in zip(timings, wavs):
      scipy.io.wavfile.write(
        str(os.path.join(dirname, filename + '_' + str(timing) + 's' + file_extension)),
        sr,
        wav,
      )


if __name__ == '__main__':
  main()