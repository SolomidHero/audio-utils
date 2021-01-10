# slash.py
# _slash_wav function from utils.py as standalone script
# apply:      to a single file
# creates:    filename_0s.wav, filename_10s.wav, ...


import argparse
import os
import librosa
from utils import _slash_wav

def get_args():
  parser = argparse.ArgumentParser(description='break a single audio file into segments')

  parser.add_argument('file', metavar='filepath', type=str, help='Path to audio file')
  parser.add_argument('--output_root', default=None, type=str,
    help='Path to store slashed audios (default: same dirictory)', required=False
  )

  parser.add_argument('--maxlen', default=10., type=float, help='Maximum length of segment (in seconds)', required=False)
  parser.add_argument('--drop_last', default=False, type=bool, help='Whether to drop last incomplete segment', required=False)

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

    # slash wav into segments
    timings, wavs = _slash_wav(y, sr, maxlen=args.maxlen, drop_last=args.drop_last, timings=True)

    # output into corresponding files
    for timing, wav in zip(timings, wavs):
      librosa.output.write_wav(
        os.path.join(dirname, filename + '_' + timing + 's' + file_extension),
        wav,
        sr,
        norm=False
      )


if __name__ == '__main__':
  main()