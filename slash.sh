#!/bin/bash
# Takes a file with timestamps ($1) and videofile ($2), output_dir (optional $3)
# breaks videofile into pieces using ffmpeg according to given timestamps
#
# Timestamps should be in the following format:
# <start_pos> <segment_length>\n
#


# https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
# didn't work for me :(

ts_file=$1
input_file=$(basename -- "$2")
output_dir=.
if [[ $# > 2 ]]; then output_dir=$3; fi

extension="${input_file#*.}"
filename="${input_file%%.*}"

while IFS="" read -r line || [ -n "${line}" ]
do
  line=(${line})
  start_pos=${line[0]}
  seg_len=${line[1]}

  if [[ ${#line[@]} == 2 ]]; then
    # evade bugs with ffmpeg waiting on input std by `echo y`

    # into videos
    # echo y | ffmpeg -hide_banner -loglevel error -ss $start_pos -i $input_file -t $seg_len -c copy $output_dir/$filename\_$start_pos\_$seg_len.$extension

    # into audios
    echo y | ffmpeg -hide_banner -loglevel error -ss $start_pos -i $input_file -t $seg_len -map a $output_dir/$filename\_$start_pos\_$seg_len.wav
  fi
done < $ts_file
