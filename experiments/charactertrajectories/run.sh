#!/bin/bash

python3 main.py --adjoint "reversible" --checkpoints "1"

checkpoints=(2 4 8 16 32 64)
for n in "${checkpoints[@]}"
do
    python3 main.py --adjoint "recursive" --checkpoints "$n"
done