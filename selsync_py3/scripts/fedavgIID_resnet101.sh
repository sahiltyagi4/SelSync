#!/bin/bash

cd ~/SelSync/selsync_py3/

worldsize=8
lr=0.1
gamma=0.1
momentum=0.9
weightdecay=0.0001
dataset='cifar10'
dir='/'
fedavgsteps=250

for rank in $(seq 1 $worldsize)
do
  procrank=$(($rank-1))
  python3 -m fedavg.iid_imgclassifier --dir=$dir --lr=$lr --gamma=$gamma --momentum=$momentum --weight-decay=$weightdecay \
  --world-size=$worldsize --rank=$procrank --dataset=$dataset --fedavg-steps=$fedavgsteps &
done