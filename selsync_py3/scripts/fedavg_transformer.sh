#!/bin/bash

cd ~/SelSync/selsync_py3/

worldsize=10
lr=2.0
stepsize=2.0
gamma=0.8
bptt=35
normclip=0.5
lrdecaysteps=2000
fedavgsteps=250
dir='/'

for rank in $(seq 1 $worldsize)
do
  procrank=$(($rank-1))
  python3 -m fedavg.launch_transformer --dir=$dir --lr=$lr --step-size=$stepsize --gamma=$gamma --bptt=$bptt \
  --world-size=$worldsize --norm-clip=$normclip --rank=$procrank --lr-decay-steps=$lrdecaysteps --fedavg-steps=$fedavgsteps &
done