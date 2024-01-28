#!/bin/bash

cd ~/SelSync/selsync_py3/

worldsize=8
lr=0.01
momentum=0.9
weightdecay=5e-4
gamma=0.1
fedavgsteps=250
dataset='cifar100'
dir='/'

for rank in $(seq 1 $worldsize)
do
  procrank=$(($rank-1))
  python3 -m fedavg.iid_imgclassifier --dir=$dir --model='vgg11' --lr=$lr --gamma=$gamma --momentum=$momentum \
  --weight-decay=$weightdecay --world-size=$worldsize --rank=$procrank --dataset=$dataset --fedavg-steps=$fedavgsteps &
done