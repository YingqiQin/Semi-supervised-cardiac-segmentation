#!/bin/bash
#SBATCH --account=def-ouriadov                  # using graham's account
#SBATCH --gpus-per-node=v100l:1            	    # request a p100 GPU (v100 gpus does not work well for this code, not sure why)
#SBATCH --mem=90G                  				# memory per node
#SBATCH --time=00-11:00                         # reserve a gpu for 15 hours.
#SBATCH --output=baseline_Unet_mixed.out               # log output

source ../../../python3.7-pytorch1.7/bin/activate
python baseline_UNet.py \
  --train_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/training \
  --valid_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/training \
  --test_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/testing \
  --train_data_list /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_training_mixed.csv \
  --valid_data_list /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_valid_mixed.csv \
  --test_data_list /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/test_subj.csv \
  --train_output_dir /home/fguo24/scratch/Reg_Seg/Baseline_Lower_mixed \
  --test_output_dir /home/fguo24/scratch/Reg_Seg/Baseline_Lower_mixed \
  --mode train