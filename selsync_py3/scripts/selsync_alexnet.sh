#!/bin/bash

cd ~/SelSync/selsync_py3/

worldsize=10
smoothmethod='custom_ewma'
windowsize=25
alpha=0.01
deltathreshold=0.3
slopewindow=50

lr=0.0001
momentum=0.9
weightdecay=5e-4
bsz=128
testbsz=128
teststeps=1171
backend='gloo'
agg='model'
datapartition='defaultpartition'
dir='/'

for rank in $(seq 1 $worldsize)
do
  procrank=$(($rank-1))
  python3 -m selsync.selsync_imgclassifier --dir=$dir --model='alexnet' --dataset='imagenet' --lr=$lr \
  --world-size=$worldsize --momentum=$momentum --weight-decay=$weightdecay --bsz=$bsz --test-bsz=$testbsz \
  --rank=$procrank --smoothing-method=$smoothmethod --windowsize=$windowsize --smoothing=$alpha \
  --delta-threshold=$deltathreshold --slope-window=$slopewindow --backend=$backend --aggregation=$agg \
  --datapartition=$datapartition --teststeps=$teststeps &
done