#!/bin/bash 
#SBATCH --job-name=jemtraining 
#SBATCH --ntasks=1 
#SBATCH --account=facepain 
#SBATCH --qos=limited 
#SBATCH --partition=ALL 
#SBATCH --cpus-per-task=2 
#SBATCH --gres=gpu:1 
#SBATCH --mem=2G 
#SBATCH --time=4-15:00:00 

echo "tmpdir for the job: "$TMPDIR 
echo "total gpu resources allocated: "$CUDA_VISIBLE_DEVICES 
echo "CPU allocated: "$(taskset -c -p $$)
pwd
python train_wrn_ebm.py --lr .0001 --dataset cifar10 --optimizer adam --p_x_weight 1.0 --p_y_given_x_weight 1.0 --p_x_y_weight 0.0 --sigma .03 --width 10 --depth 28 --save_dir //nas/home/adani/projects/adversarial/apurva_jem/JEM/outdir --plot_uncond --warmup_iters 1000