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
declare -a sample_complexity=(2.0)
declare -a n_features=(1024)
declare -a n_hidden=(1 2)
declare -a lr=(0.002)
declare -a n_epochs=(2000)
declare -a momentum=(0.9)
declare -a batch_size=(128)
declare -a optimizer=("p-sgd")
declare -a tau=(0.5 2.0)
for ((s=0;s<${#manual_seed[@]};++s)); do
for ((a=0;a<${#sample_complexity[@]};++a)); do
for ((f=0;f<${#n_features[@]};++f)); do
for ((h=0;h<${#n_hidden[@]};++h)); do
for ((l=0;l<${#lr[@]};++l)); do
for ((m=0;m<${#momentum[@]};++m)); do
for ((o=0;o<${#optimizer[@]};++o)); do
for ((b=0;b<${#batch_size[@]};++b)); do
for ((t=0;t<${#tau[@]};++t)); do
srun python -u /home/dmitriev/spoc_master_project/main.py --root_dir psgd_1024 --start_seed ${manual_seed[s]} \
 --batch_size ${batch_size[b]} --sample_complexity ${sample_complexity[a]} --n_test 100 \
 --n_features ${n_features[f]} --n_hidden ${n_hidden[h]} --momentum_factor ${momentum[m]} \
 --verbose_freq 100 --optimizer ${optimizer[o]} --n_epochs ${n_epochs[l]} --lr ${lr[l]} --train_threshold 1e-8 \
 --gradient_threshold 5e-5 --test_threshold 1e-3 --n_runs 10 --persistence_time ${tau[t]}
done
done
done
done
done
done
done
done
done
