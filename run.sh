#!/bin/bash

declare -a manual_seed=(22)
declare -a sample_complexity=(5.0)
declare -a n_features=(1024)
declare -a n_hidden=(1)
declare -a lr=(0.001)
declare -a n_epochs=(8000)
declare -a momentum=(0.0)
declare -a batch_size=(1)
declare -a optimizer=("gd")
for ((i=0;i<${#manual_seed[@]};++i)); do
for ((j=0;j<${#sample_complexity[@]};++j)); do
for ((k=0;k<${#n_features[@]};++k)); do
for ((l=0;l<${#n_hidden[@]};++l)); do
for ((m=0;m<${#lr[@]};++m)); do
for ((n=0;n<${#momentum[@]};++n)); do
for ((p=0;p<${#optimizer[@]};++p)); do
for ((t=0;t<${#batch_size[@]};++t)); do
python -u ./main.py --root_dir sgd --start_seed ${manual_seed[i]} \
 --batch_size ${batch_size[t]} --sample_complexity ${sample_complexity[j]} \
 --n_features ${n_features[k]} --n_hidden ${n_hidden[l]} --momentum_factor ${momentum[n]} \
 --verbose_freq 5 --optimizer ${optimizer[t]} --n_epochs ${n_epochs[m]} --lr ${lr[m]} --threshold 1e-5
done
done
done
done
done
done
done
done
