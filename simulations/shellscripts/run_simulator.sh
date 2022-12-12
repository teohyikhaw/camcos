#!/bin/bash

#SBATCH -o /home/yteoh/results/slurm-%j.out # STDOUT
#SBATCH --nodes=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yikhaw.teoh@sjsu.edu

module purge
module load intel-python3 openmpi-3.0.1/intel intel
source activate p310
export PYTHONPATH=/home/yteoh/camcos/

call_data_standard_value=25
call_data_learning_rate=1/8
call_data_target_limit=25000
step_count=400
directory="new_results"

call_data_learning_rates=("1/13" "1/12" "1/11" "1/10" "1/9")
call_data_target_limits=("21500" "22000" "22500" "23000" "23500" "24000" "24500" "25000")

for i in "${call_data_learning_rates[@]}"
do
  for j in "${call_data_target_limits[@]}"
  do
  srun --ntasks 8 python /home/yteoh/camcos/simulations/shellscripts/single_trajectory_simulation.py $call_data_standard_value "$i" "$j" $step_count $directory
  srun --ntasks 1 python /home/yteoh/camcos/simulations/shellscripts/average_trajectories.py $call_data_standard_value "$i" "$j" $step_count $directory
  done
done