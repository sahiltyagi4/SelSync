#!/bin/bash

cd ~/SelSync/selsync_py3/

worldsize=10
lr=0.01
momentum=0.9
weightdecay=5e-4
gamma=0.1
smoothmethod='custom_ewma'
windowsize=25
alpha=0.01
deltathreshold=0.3
slopewindow=100
dataset='cifar100'
backend='gloo'
agg='model'
datapartition='selsyncpartition'
dir='/'
# does data have iid or non-iid distribution
datadist='iid'
# not performing data injection, else 1 if enabling it
datainjection=0
# fraction of workers to use
alpha=0.25
# fraction of batch-size to use
beta=0.25

for rank in $(seq 1 $worldsize)
do
  procrank=$(($rank-1))
  echo '###### going to launch training for rank '$procrank
  python3 -m selsync.selsync_imgclassifier --dir=$dir --model='vgg11' --lr=$lr --gamma=$gamma --momentum=$momentum \
  --weight-decay=$weightdecay --world-size=$worldsize --rank=$procrank --dataset=$dataset --smoothing-method=$smoothmethod \
  --windowsize=$windowsize --smoothing=$alpha --delta-threshold=$deltathreshold --slope-window=$slopewindow \
  --backend=$backend --aggregation=$agg --datapartition=$datapartition --data-dist=$datadist \
  --data-injection=$datainjection --alpha=$alpha --beta=$beta &
done