#!/bin/bash

export JULIA_NUM_THREADS=1

SEEDS=$(seq 7000 8000)
BETAS="3.0 6.0 9.0"
JOBS=10

mkdir -p logs
rm -rf logs/*

parallel -j $JOBS '
mkdir -p logs/beta_{2} &&
julia pigsfli.jl \
    -D 2 --Lx 3 --Ly 3 -N 9 \
    --geometry "triangular" \
    --boundary "obc" \
    -U 16.0 --mu 4.0 --eta 0.01 \
    --seed {1} \
    --bin-size 1000 \
    --bins-wanted 1000 \
    --beta {2} \
    --get-density \
    --get-corr-mat \
    --trial-state gutzwiller \
    --kappa 1.189 \
    -Z 40.0 \
    > logs/beta_{2}/seed_{1}.log 2>&1
' ::: $SEEDS ::: $BETAS
