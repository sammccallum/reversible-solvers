#!/bin/bash

checkpoints=(2 4 8 16 32 44)
for repeat in {1..2}
do
    echo "Repeat $repeat"
    python3 white_dwarf.py --adjoint reversible --checkpoints "1" --key "$repeat"

    for n in "${checkpoints[@]}"
    do
        python3 white_dwarf.py --adjoint recursive --checkpoints "$n" --key "$repeat"
    done
done