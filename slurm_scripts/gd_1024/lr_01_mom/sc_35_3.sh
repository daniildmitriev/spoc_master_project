#!/bin/bash
#SBATCH --chdir /home/dmitriev/spoc_master_project/slurm_output
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 4096
#SBATCH --time 30:00:00
#SBATCH -o testres.out
#SBATCH -e error.out
module load intel
module load python

declare -a manual_seed=(32)
declare -a sample_complexity=(3.5)
declare -a n_features=(1024)
declare -a n_hidden=(1 2 5 10)
declare -a lr=(0.01)
declare -a n_epochs=(10000)
declare -a momentum=(0.0)
declare -a batch_size=(8)
declare -a optimizer=("gd")
for ((i=0;i<${#manual_seed[@]};++i)); do
for ((j=0;j<${#sample_complexity[@]};++j)); do
for ((k=0;k<${#n_features[@]};++k)); do
for ((l=0;l<${#n_hidden[@]};++l)); do
for ((m=0;m<${#lr[@]};++m)); do
for ((n=0;n<${#momentum[@]};++n)); do
for ((p=0;p<${#optimizer[@]};++p)); do
for ((t=0;t<${#batch_size[@]};++t)); do
srun python -u /home/dmitriev/spoc_master_project/main.py --root_dir gd_1024 --start_seed ${manual_seed[i]} \
 --batch_size ${batch_size[t]} --sample_complexity ${sample_complexity[j]} --n_test 100 \
 --n_features ${n_features[k]} --n_hidden ${n_hidden[l]} --momentum_factor ${momentum[n]} \
 --verbose_freq 100 --optimizer ${optimizer[p]} --n_epochs ${n_epochs[m]} --lr ${lr[m]} --threshold 1e-5 \
 --gradient_threshold 1e-3 --test_threshold 1e-3 --n_runs 5
done
done
done
done
done
done
done
done
