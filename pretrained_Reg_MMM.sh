#!/bin/bash
#SBATCH --account=def-ouriadov                  # using graham's account
#SBATCH --gpus-per-node=p100:1            	    # request a p100 GPU (v100 gpus does not work well for this code, not sure why)
#SBATCH --mem=90G                  				# memory per node
#SBATCH --time=00-23:00                         # reserve a gpu for 15 hours.
#SBATCH --output=Pretrained_Reg_MMM.out               # log output

source ../../../python3.7-pytorch1.7/bin/activate
python RegNet.py \
    --train_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/MMM_chall_1/MMM/Converted_Labeled \
    --train_data_list_A /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_training_MMM.csv \
    --train_data_list_B /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_training_MMM.csv \
    --valid_data_list_A /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_training_MMM.csv \
    --valid_data_list_B /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_training_MMM.csv \
    --model_dir /home/fguo24/scratch/Reg_Seg/Pretrained_Reg_MMM/Checkpoint_Pretrained_Reg \
    --log_dir  /home/fguo24/scratch/Reg_Seg/Pretrained_Reg_MMM/Log_Pretrained_Reg \
    --train_output_dir /home/fguo24/scratch/Reg_Seg/Pretrained_Reg_MMM/Result_Pretrained \
    --pattern train \
    --checkpoint_path /home/fguo24/scratch/Reg_Seg/Pretrained_Reg/Checkpoint_Pretrained_Reg/trained_model.pth \
    --test_data_list_A /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/test_subj_1.csv \
    --test_data_list_B /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/test_subj_2.csv \
    --test_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/testing
