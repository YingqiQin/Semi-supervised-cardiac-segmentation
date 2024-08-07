#!/bin/bash
#SBATCH --account=def-ouriadov                  # using graham's account
#SBATCH --gpus-per-node=v100l:1            	    # request a p100 GPU (v100 gpus does not work well for this code, not sure why)
#SBATCH --mem=90G                  				# memory per node
#SBATCH --time=00-11:00                         # reserve a gpu for 15 hours.
#SBATCH --output=Ablation_2Seg_noaxu_50par_mixed.out               # log output

source ../../../python3.7-pytorch1.7/bin/activate
python train_2Seg_no_auxloss.py \
  --train_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/training \
  --train_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_training_mixed.csv \
  --train_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_training_mixed.csv \
  --valid_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_valid_mixed.csv \
  --valid_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_valid_mixed.csv \
  --resume /home/fguo24/scratch/Reg_Seg/Pretrained_Reg/Checkpoint_Pretrained_Reg/trained_model.pth \
  --train_output_dir /home/fguo24/scratch/Reg_Seg/Ablation_2Seg_noaux_50par_mixed \
  --test_output_dir /home/fguo24/scratch/Reg_Seg/Ablation_2Seg_noaux_50par_mixed \
  --test_data_list /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/test_subj.csv \
  --test_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/testing \
  --mode train \
  --epochs 250 \
  --learning_rate 4e-4
python train_2Seg_no_auxloss.py \
  --train_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/training \
  --train_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_training_mixed.csv \
  --train_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_training_mixed.csv \
  --valid_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_valid_mixed.csv \
  --valid_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_valid_mixed.csv \
  --resume /home/fguo24/scratch/Reg_Seg/Pretrained_Reg/Checkpoint_Pretrained_Reg/trained_model.pth \
  --train_output_dir /home/fguo24/scratch/Reg_Seg/Ablation_2Seg_noaux_50par_mixed \
  --test_output_dir /home/fguo24/scratch/Reg_Seg/Ablation_2Seg_noaux_50par_mixed \
  --test_data_list /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/test_subj.csv \
  --test_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/testing \
  --mode test_reg \
  --epochs 250 \
  --learning_rate 4e-4