#!/bin/bash

n=2
python3 complexity.py --adjoint reversible --checkpoints "$n"

checkpoints=(2 4 8)
for n in "${checkpoints[@]}"
do
    python3 complexity.py --adjoint recursive --checkpoints "$n"
done