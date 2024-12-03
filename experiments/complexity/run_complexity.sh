#!/bin/bash

# n=2
# python3 complexity.py --adjoint reversible --checkpoints "$n"

for n in {30..50}
do
    python3 complexity.py --adjoint recursive --checkpoints "$n"
done