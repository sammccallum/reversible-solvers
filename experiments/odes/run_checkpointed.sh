#!/bin/bash

checkpoints=(2 4 8 16 32 44)
repeats=4
for i in $(seq 1 $repeats); do
  echo "Iteration $i"
  for n in "${checkpoints[@]}"
    do
        python3 pendulum.py --checkpoints "$n"
    done
done
