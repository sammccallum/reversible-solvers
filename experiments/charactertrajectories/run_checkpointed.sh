#!/bin/bash

checkpoints=(2 4 8 16 32 64)
for n in "${checkpoints[@]}"
do
    python3 main.py --checkpoints "$n"
done