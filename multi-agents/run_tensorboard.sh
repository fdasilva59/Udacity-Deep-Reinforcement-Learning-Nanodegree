#!/bin/bash
tensorboard --logdir=./log/ --host=0.0.0.0 --port=6006 &> /dev/null &
echo "Tensorboard is running on port 6006"
#echo "Wait around 10 seconds and open this link at your browser (ignore other outputs):"
#echo "https://$WORKSPACEID-3000.$WORKSPACEDOMAIN"
