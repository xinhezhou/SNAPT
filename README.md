# SNAPT

Anonymous repository for SNAPT, part of submission to evoRL workshop at GECCO 2021.

Folder C4 contains Connect Four related code, folder SNAPT contains SNAPT related code.

AlphaZero code is taken from 

https://github.com/suragnair/alpha-zero-general


# Usage

The SNAPT directory contains all files related to SNAPT

The ConnectFour directory contains all files related to Connect Four

In each directory, there is a "training" notebook and an "arena" notebook

The training notebook allows for neural networks training on that game using the methods from the paper.
  - The trained networks are saved in the checkpoint directory

The arena notebook allows for comparison of trained networks
  - The trained networks are loaded from the checkpoint directory
