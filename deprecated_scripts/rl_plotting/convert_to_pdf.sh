#!/bin/bash

directory=$1
prefix=$2

dsize=0.5
gpu1=8
gpu2=32
bw1=25
bw2=25000000
app1="cholesky"
app2="sweeps"

python3 pdfs_to_ppts.py $prefix "random" $dsize $gpu1 $gpu2 $bw1 $bw2 $app1 $directory/random_outputs/pdfs 
python3 pdfs_to_ppts.py $prefix "heft" $dsize $gpu1 $gpu2 $bw1 $bw2 $app1 $directory/heft_outputs/pdfs 
python3 pdfs_to_ppts.py $prefix "random" $dsize $gpu1 $gpu2 $bw1 $bw2 $app2 $directory/random_outputs/pdfs 
python3 pdfs_to_ppts.py $prefix "heft" $dsize $gpu1 $gpu2 $bw1 $bw2 $app2 $directory/heft_outputs/pdfs 
