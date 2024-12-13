#!/bin/bash

python3 lorenz.py --adjoint reversible --checkpoints "1"

checkpoints=(2 4 8 16 32 44)
for n in "${checkpoints[@]}"
do
    python3 lorenz.py --adjoint recursive --checkpoints "$n"
done
