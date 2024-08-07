#!/bin/bash
#SBATCH --account=def-ouriadov                  # using graham's account
#SBATCH --gpus-per-node=p100:1            	    # request a p100 GPU (v100 gpus does not work well for this code, not sure why)
#SBATCH --mem=90G                  				# memory per node
#SBATCH --time=00-23:00                         # reserve a gpu for 15 hours.
#SBATCH --output=Pretrained_Reg_series.out               # log output

source ../../../python3.7-pytorch1.7/bin/activate
python RegNet_series.py \
    --train_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/training \
    --all_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/10par_all_phase.csv \
    --val_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/10par_all_phases_val.csv \
    --model_dir /home/fguo24/scratch/Reg_Seg/Pretrained_Reg/Checkpoint_Pretrained_Reg_series \
    --log_dir  /home/fguo24/scratch/Reg_Seg/Pretrained_Reg/Log_Pretrained_Reg_series \
    --train_output_dir /home/fguo24/scratch/Reg_Seg/Pretrained_Reg/Result_Pretrained_series \
    --pattern train \
    --checkpoint_path /home/fguo24/scratch/Reg_Seg/Pretrained_Reg/Checkpoint_Pretrained_Reg_series/trained_model.pth \
    --test_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/testing