#!/bin/bash
export PYTHONPATH=/mnt/d/Monaf/Personal/Time_series_forecasting/T3Time:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

data_path="ETTh1"
seq_len=96
batch_size=16

# pred_len 96
pred_len=96
learning_rate=1e-4
channel=256
e_layer=1
d_layer=1
dropout_n=0.4

log_path="./Results/${data_path}/"
mkdir -p $log_path
log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size 256 \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len 96 \
  --epochs 150 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &

# pred_len = 192
pred_len=192
learning_rate=1e-4
channel=256
e_layer=1
d_layer=2
dropout_n=0.6

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size 32 \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 150 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &

# pred_len 336
pred_len=336
learning_rate=1e-4
channel=64
dropout_n=0.7
e_layer=1
d_layer=2

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 120 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &

# pred_len 720
pred_len=720
learning_rate=1e-4
channel=64
dropout_n=0.5
e_layer=3
d_layer=4

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size 32 \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 150 \
  --seed 2024 \
  --channel $channel \
  --head 8 \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &