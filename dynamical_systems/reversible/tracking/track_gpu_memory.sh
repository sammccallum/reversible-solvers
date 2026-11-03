#!/bin/bash

peak_memory=0

save_peak_memory() {
  echo "$peak_memory"
  exit 0
}

# Runs on CTRL-C
trap save_peak_memory SIGINT

while true; do
  current_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -n1)

  if (( current_memory > peak_memory )); then
    peak_memory=$current_memory
  fi

  # Measure every 0.1 seconds
  sleep 0.1
done