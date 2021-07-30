#!/bin/bash
#SBATCH --chdir /home/dmitriev/spoc_master_project/slurm_output
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --mem 4096
#SBATCH --time 20:00:00
#SBATCH -o testres.out
#SBATCH -e error.out
module load intel
module load python

declare -a manual_seed=(2)
declare -a n_samples=(128)
declare -a n_features=(32)
declare -a n_hidden=(1 10 32)
declare -a lr=(0.002 0.0005)
declare -a n_epochs=(10000 20000)
for ((i=0;i<${#manual_seed[@]};++i)); do
for ((j=0;j<${#n_samples[@]};++j)); do
for ((k=0;k<${#n_features[@]};++k)); do
for ((l=0;l<${#n_hidden[@]};++l)); do
for ((m=0;m<${#lr[@]};++m)); do
srun python -u /home/dmitriev/spoc_master_project/main.py \
 --n_train ${n_samples[j]} --n_test ${n_samples[j]} --n_features ${n_features[k]} --n_hidden ${n_hidden[l]} \
 --verbose_freq 100 --optimizer sgd --n_epochs ${lr[m]} --lr ${lr[m]} --threshold 1e-8
done
done
done
done
done
