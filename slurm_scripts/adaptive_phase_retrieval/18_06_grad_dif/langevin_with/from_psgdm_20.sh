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

declare -a manual_seed=(42)
declare -a sample_complexity=(2.0)
declare -a n_features=(1024)
declare -a n_hidden=(1)
declare -a lr=(0.005)
declare -a n_epochs=(10000)
declare -a momentum=(0.9)
declare -a batch_size=(1)
declare -a optimizer=("langevin")
declare -a tau=(2)
declare -a K=(0.67449)
declare -a activation=("quadratic")
declare -a second_layer_activation=("none")
declare -a loss=("mse")
for ((s=0;s<${#manual_seed[@]};++s)); do
for ((a=0;a<${#sample_complexity[@]};++a)); do
for ((f=0;f<${#n_features[@]};++f)); do
for ((h=0;h<${#n_hidden[@]};++h)); do
for ((l=0;l<${#lr[@]};++l)); do
for ((m=0;m<${#momentum[@]};++m)); do
for ((o=0;o<${#optimizer[@]};++o)); do
for ((b=0;b<${#batch_size[@]};++b)); do
for ((t=0;t<${#tau[@]};++t)); do
for ((k=0;k<${#K[@]};++k)); do
for ((c=0;c<${#activation[@]};++c)); do
for ((q=0;q<${#loss[@]};++q)); do
srun python -u /home/dmitriev/spoc_master_project/main.py \
 --root_dir 19_06_langevin \
 --noise_std_dir 18_06_grad_dif/1624184039_optim_p-sgd_labels_quadratic_loss_mse_samplecomplexity_2.0_nfeatures_1024_nhidden_1_lr_0.005_momentum_0.9_nesterov_False \
 --labels quadratic \
 --start_seed ${manual_seed[s]} \
 --psgd_adaptive_bs True \
 --batch_size ${batch_size[b]} \
 --sample_complexity ${sample_complexity[a]} \
 --n_test 100 \
 --early_stopping_epochs -1 \
 --n_features ${n_features[f]} \
 --n_hidden ${n_hidden[h]} \
 --momentum_factor ${momentum[m]} \
 --verbose_freq 100 \
 --optimizer ${optimizer[o]} \
 --n_epochs ${n_epochs[l]} \
 --lr ${lr[l]} \
 --train_threshold 1e-8 \
 --project_on_sphere True \
 --gradient_threshold 5e-10 \
 --test_threshold 1e-8 \
 --compute_grad_dif False \
 --n_runs 10 --fix_teacher_change_data True --reverse_mult_by_sqrt False \
 --persistence_time ${tau[t]} \
 --use_nesterov False \
 --loss ${loss[q]} \
 --activation ${activation[c]} \
 --second_layer_activation ${second_layer_activation[c]} \
 --symmetric_door_channel_K ${K[k]}
done
done
done
done
done
done
done
done
done
done
done
done
