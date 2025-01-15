#!/bin/bash

checkpoints=(32 16 8 4 2)
for repeat in {0..1}
do
    echo "Repeat $repeat"
    python3 pendulum.py --adjoint reversible --checkpoints "1" --key "$repeat"

    for n in "${checkpoints[@]}"
    do
        python3 pendulum.py --adjoint recursive --checkpoints "$n" --key "$repeat"
    done
done