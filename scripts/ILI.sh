#!/bin/bash
export PYTHONPATH=/mnt/d/Monaf/Personal/Time_series_forecasting/T3Time:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

data_path="ILI"
seq_len=36
batch_size=16

# pred_len = 24
pred_len=24
learning_rate=1e-4
channel=32
e_layer=1
d_layer=1
dropout_n=0.1

log_path="./Results/${data_path}/"
mkdir -p $log_path
log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 100 \
  --seed 6666 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --num_workers 10 \
  --d_llm 768 > $log_file &

# pred_len = 36
pred_len=36
learning_rate=1e-4
channel=64
e_layer=1
d_layer=1
dropout_n=0.1

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 150 \
  --seed 6666 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &

# pred_len = 48
pred_len=48
learning_rate=2.5e-3
channel=128
e_layer=1
d_layer=1
dropout_n=0.5
batch_size=32

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 150 \
  --seed 8888 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &

# pred_len = 60
pred_len=60
learning_rate=1e-4
channel=64
e_layer=1
d_layer=1
dropout_n=0.1
batch_size=32

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 150 \
  --seed 6666 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &