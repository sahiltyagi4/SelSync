#!/bin/bash

cd ~/SelSync/selsync_py3/

worldsize=8
smoothmethod='custom_ewma'
windowsize=25
alpha=0.01
deltathreshold=0.3
slopewindow=50

lr=2.0
stepsize=2.0
gamma=0.8
bptt=35
normclip=0.5
lrdecaysteps=2000
dir='/'
backend='gloo'
aggregation='model'
datapartition='defaultpartition'

for rank in $(seq 1 $worldsize)
do
  procrank=$(($rank-1))
  python3 -m selsync.selsync_transformer --dir=$dir --lr=$lr --step-size=$stepsize --gamma=$gamma --bptt=$bptt \
  --world-size=$worldsize --norm-clip=$normclip --rank=$procrank --lr-decay-steps=$lrdecaysteps \
  --smoothing-method=$smoothmethod --windowsize=$windowsize --smoothing=$alpha --slope-window=$slopewindow \
  --delta-threshold=$deltathreshold --backend=$backend --aggregation=$aggregation --datapartition=$datapartition &
done