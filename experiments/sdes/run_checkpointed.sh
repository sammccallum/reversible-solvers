#!/bin/bash


checkpoints=(2 4 8 16)
for n in "${checkpoints[@]}"
do
    python3 main.py --adjoint recursive --checkpoints "$n"
done
