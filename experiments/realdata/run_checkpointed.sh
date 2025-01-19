#!/bin/bash

checkpoints=(31 16 8 4 2)
for repeat in {0..2}
do
    echo "Repeat $repeat"
    python3 main.py --adjoint reversible --checkpoints "1" --key "$repeat"

    for n in "${checkpoints[@]}"
    do
        python3 main.py --adjoint recursive --checkpoints "$n" --key "$repeat"
    done
done