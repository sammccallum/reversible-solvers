#!/bin/bash

checkpoints=(2 4 8 16 32)
# for n in "${checkpoints[@]}"
# do
#     python3 white_dwarf.py --checkpoints "$n"
# done

for n in "${checkpoints[@]}"
do
    python3 fitzhugh_nagumo.py --checkpoints "$n"
done