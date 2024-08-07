#!/bin/bash
#SBATCH --account=def-ouriadov                  # using graham's account
#SBATCH --gpus-per-node=v100l:1            	    # request a p100 GPU (v100 gpus does not work well for this code, not sure why)
#SBATCH --mem=90G                  				# memory per node
#SBATCH --time=00-11:00                         # reserve a gpu for 15 hours.
#SBATCH --output=DeepAtlas_MMM.out               # log output

source ../../../python3.7-pytorch1.7/bin/activate
python DeepAtlas.py \
  --train_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/MMM_chall_1/MMM/Converted_Labeled \
  --train_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_training_MMM.csv \
  --train_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_training_MMM.csv \
  --valid_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_training_MMM.csv \
  --valid_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_training_MMM.csv \
  --resume /home/fguo24/scratch/Reg_Seg/Pretrained_Reg_MMM/Checkpoint_Pretrained_Reg/trained_model.pth \
  --train_output_dir /home/fguo24/scratch/Reg_Seg/DeepAtlas_50par_MMM \
  --test_output_dir /home/fguo24/scratch/Reg_Seg/DeepAtlas_50par_MMM \
  --test_data_list /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/test_subj_MMM.csv \
  --test_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/MMM_chall_1/MMM/Converted_Labeled \
  --mode train \
  --epochs 200 \
  --learning_rate 1e-4 \
  --reg_batch_size 1