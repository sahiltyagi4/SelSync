#!/bin/bash

cd ~/SelSync/selsync_py3/

worldsize=8
lr=0.0001
momentum=0.9
weightdecay=5e-4
bsz=128
testbsz=128
teststeps=1171
fedavgsteps=250
dir='/'

for rank in $(seq 1 $worldsize)
do
  procrank=$(($rank-1))
  python3 -m fedavg.iid_imgclassifier --dir=$dir --model='alexnet' --dataset='imagenet' --lr=$lr --teststeps=$teststeps \
  --world-size=$worldsize --momentum=$momentum --weight-decay=$weightdecay --bsz=$bsz --test-bsz=$testbsz \
  --rank=$procrank --fedavg-steps=$fedavgsteps &
done