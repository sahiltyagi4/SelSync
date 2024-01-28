#!/bin/bash

cd ~/SelSync/selsync_py3/

worldsize=10
lr=0.01
momentum=0.9
weightdecay=5e-4
gamma=0.1
fedavgsteps=250

model='vgg11'
dataset='cifar100'
backend='gloo'
dir='/'

for rank in $(seq 1 $worldsize)
do
  procrank=$(($rank-1))
  python3 -m fedavg.noniid_imgclassifier --dir=$dir --model=$model --lr=$lr --gamma=$gamma --momentum=$momentum \
  --weight-decay=$weightdecay --world-size=$worldsize --rank=$procrank --dataset=$dataset --fedavg-steps=$fedavgsteps \
  --backend=$backend &
done