#!/bin/bash

cd ~/SelSync/selsync_py3/

worldsize=10
lr=0.1
gamma=0.1
momentum=0.9
weightdecay=0.0001
dataset='cifar10'
model='resnet101'
backend='nccl'
dir='/'
fedavgsteps=250

for rank in $(seq 1 $worldsize)
do
  procrank=$(($rank-1))
  python3 -m fedavg.noniid_imgclassifier --dir=$dir --lr=$lr --gamma=$gamma --momentum=$momentum --weight-decay=$weightdecay \
  --world-size=$worldsize --rank=$procrank --dataset=$dataset --fedavg-steps=$fedavgsteps --model=$model \
  --backend=$backend &
done