#!/bin/bash
#SBATCH --account=def-ouriadov                  # using graham's account
#SBATCH --gpus-per-node=v100l:1            	    # request a p100 GPU (v100 gpus does not work well for this code, not sure why)
#SBATCH --mem=90G                  				# memory per node
#SBATCH --time=00-23:00                         # reserve a gpu for 15 hours.
#SBATCH --output=semi_2Seg_TAtten_features_EDlabeled.out               # log output

source ../../../python3.7-pytorch1.7/bin/activate
python train_2Seg_TAtten_features.py \
  --train_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/training \
  --train_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_training_ED.csv \
  --train_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_training_ED.csv \
  --valid_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_valid_ED.csv \
  --valid_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_valid_ED.csv \
  --resume /home/fguo24/scratch/Reg_Seg/Pretrained_Reg/Checkpoint_Pretrained_Reg/trained_model.pth \
  --TPer_state_dict_resume /home/fguo24/scratch/Reg_Seg/TAtten/model/TPer.pth \
  --train_output_dir /home/fguo24/scratch/Reg_Seg/semi_2Seg_TAtten_features_EDlabeled \
  --test_output_dir /home/fguo24/scratch/Reg_Seg/semi_2Seg_TAtten_features_EDlabeled \
  --test_data_list /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/test_subj.csv \
  --test_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/testing \
  --mode train \
  --epochs 250 \
  --learning_rate 4e-4