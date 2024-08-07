#!/bin/bash
#SBATCH --account=def-ouriadov                  # using graham's account
#SBATCH --gpus-per-node=v100l:1            	    # request a p100 GPU (v100 gpus does not work well for this code, not sure why)
#SBATCH --mem=90G                  				# memory per node
#SBATCH --time=00-11:00                         # reserve a gpu for 15 hours.
#SBATCH --output=semi_2Seg_TAtten_features_10par_MMM.out               # log output

source ../../../python3.7-pytorch1.7/bin/activate
python train_2Seg_TAtten_features.py \
  --train_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/MMM_chall_1/MMM/Converted_Labeled \
  --train_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/10par_supervised_training_MMM.csv \
  --train_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/10par_unsupervised_training_MMM.csv \
  --valid_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/10par_supervised_training_MMM.csv \
  --valid_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/10par_unsupervised_training_MMM.csv \
  --resume /home/fguo24/scratch/Reg_Seg/Pretrained_Reg_MMM/Checkpoint_Pretrained_Reg/trained_model.pth \
  --TPer_state_dict_resume /home/fguo24/scratch/Reg_Seg/TAtten_MMM/model/TPer.pth \
  --train_output_dir /home/fguo24/scratch/Reg_Seg/semi_2Seg_TAtten_features_10par_MMM \
  --test_output_dir /home/fguo24/scratch/Reg_Seg/semi_2Seg_TAtten_features_10par_MMM \
  --test_data_list /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/test_subj_MMM.csv \
  --test_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/MMM_chall_1/MMM/Converted_Labeled \
  --mode train \
  --epochs 250 \
  --learning_rate 4e-4