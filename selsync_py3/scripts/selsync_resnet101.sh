#!/bin/bash

cd ~/SelSync/selsync_py3/

smoothmethod='custom_ewma'
windowsize=25
smoothing=0.01
deltathreshold=0.3
slopewindow=100
worldsize=10
lr=0.1
gamma=0.1
momentum=0.9
weightdecay=0.0001
dataset='cifar10'
backend='gloo'
agg='parameter'
bsz=32
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
  python3 -m selsync.selsync_imgclassifier --dir=$dir --lr=$lr --gamma=$gamma --momentum=$momentum \
  --world-size=$worldsize --rank=$procrank --dataset=$dataset --smoothing-method=$smoothmethod --windowsize=$windowsize \
  --smoothing=$smoothing --delta-threshold=$deltathreshold --slope-window=$slopewindow --backend=$backend \
  --aggregation=$agg --datapartition=$datapartition --bsz=$bsz --data-dist=$datadist --data-injection=$datainjection \
  --alpha=$alpha --beta=$beta --weight-decay=$weightdecay &
done