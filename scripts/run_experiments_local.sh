#!/bin/bash

lrs=("0.001" "0.0003")
lr_names=("fast" "slow")
dropouts=("0" "0.1" "0.2" "0.3")
dropout_names=("0" "1" "2" "3")

for ((i=0; i < ${#lrs[@]}; i++))
do
  for ((j=0; j < ${#dropouts[@]}; j++))
  do
  cnn_save_path="saved_models/cnn_lr_${lr_names[$i]}_dr_${dropout_names[$j]}"
  lstm_save_path="saved_models/lstm_lr_${lr_names[$i]}_dr_${dropout_names[$j]}"
  echo $cnn_save_path
  python scripts/main.py "--train" "--model_name" "cnn" "--save_path" ${cnn_save_path} "--lr" ${lrs[$i]} "--dropout" ${dropouts[$j]} "--training_data_size" "100"
  echo $lstm_save_path
  python scripts/main.py "--train" "--model_name" "lstm" "--save_path" ${lstm_save_path} "--lr" ${lrs[$i]} "--dropout" ${dropouts[$j]} "--training_data_size" "100"
  done
done
