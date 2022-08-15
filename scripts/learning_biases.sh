#!/bin/sh

python3 learning.py --lr=0.01 --weight_decay=0 --biases_only --optim=AdamW --batch_size=64 --epochs=5000 --split=$SPLIT &
python3 learning.py --lr=0.1 --weight_decay=0 --biases_only --optim=AdamW --batch_size=64 --epochs=5000 --split=$SPLIT &
python3 learning.py --lr=1 --weight_decay=0 --biases_only --optim=AdamW --batch_size=64 --epochs=5000 --split=$SPLIT &
python3 learning.py --lr=10 --weight_decay=0 --biases_only --optim=AdamW --batch_size=64 --epochs=5000 --split=$SPLIT &
python3 learning.py --lr=100 --weight_decay=0 --biases_only --optim=AdamW --batch_size=64 --epochs=5000 --split=$SPLIT &
