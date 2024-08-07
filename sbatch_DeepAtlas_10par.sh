#!/bin/bash
#SBATCH --account=def-ouriadov                  # using graham's account
#SBATCH --gpus-per-node=v100l:1            	    # request a p100 GPU (v100 gpus does not work well for this code, not sure why)
#SBATCH --mem=90G                  				# memory per node
#SBATCH --time=00-11:00                         # reserve a gpu for 15 hours.
#SBATCH --output=DeepAtlas_10par.out               # log output

source ../../../python3.7-pytorch1.7/bin/activate
python DeepAtlas.py \
  --train_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/training \
  --train_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/10par_supervised_training.csv \
  --train_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/10par_unsupervised_training.csv \
  --valid_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/10par_supervised_valid.csv \
  --valid_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/10par_unsupervised_valid.csv \
  --resume /home/fguo24/scratch/Reg_Seg/Pretrained_Reg/Checkpoint_Pretrained_Reg/trained_model.pth \
  --train_output_dir /home/fguo24/scratch/Reg_Seg/DeepAtlas_10par \
  --test_output_dir /home/fguo24/scratch/Reg_Seg/DeepAtlas_10par \
  --test_data_list /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/test_subj.csv \
  --test_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/testing \
  --mode train \
  --epochs 200 \
  --learning_rate 1e-4 \
  --reg_batch_size 1