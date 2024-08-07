#!/bin/bash
#SBATCH --account=def-ouriadov                  # using graham's account
#SBATCH --gpus-per-node=v100l:1            	    # request a p100 GPU (v100 gpus does not work well for this code, not sure why)
#SBATCH --mem=90G                  				# memory per node
#SBATCH --time=00-11:00                         # reserve a gpu for 15 hours.
#SBATCH --output=DeepAtlas_Philips.out               # log output

source ../../../python3.7-pytorch1.7/bin/activate
python DeepAtlas.py \
  --train_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/MMM_chall_1/MMM/Converted_Labeled \
  --train_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_training_Philips.csv \
  --train_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_training_Philips.csv \
  --valid_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_training_Philips.csv \
  --resume /home/fguo24/scratch/Reg_Seg/Pretrained_Reg_Philips/Checkpoint_Pretrained_Reg/trained_model.pth \
  --valid_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_training_Philips.csv \
  --train_output_dir /home/fguo24/scratch/Reg_Seg/DeepAtlas_50par_Philips \
  --test_output_dir /home/fguo24/scratch/Reg_Seg/DeepAtlas_50par_Philips \
  --test_data_list /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/test_subj_Philips.csv \
  --test_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/MMM_chall_1/MMM/Converted_Labeled \
  --mode train \
  --epochs 250 \
  --learning_rate 1e-4 \
  --reg_batch_size 1