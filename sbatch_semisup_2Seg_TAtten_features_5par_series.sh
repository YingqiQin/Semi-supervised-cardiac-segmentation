#!/bin/bash
#SBATCH --account=def-ouriadov                  # using graham's account
#SBATCH --gpus-per-node=v100l:1            	    # request a p100 GPU (v100 gpus does not work well for this code, not sure why)
#SBATCH --mem=90G                  				# memory per node
#SBATCH --time=01-02:00                         # reserve a gpu for 15 hours.
#SBATCH --output=semi_2Seg_TAtten_features_5par_series.out               # log output

source ../../../python3.7-pytorch1.7/bin/activate
python train_2Seg_TAtten_features_series.py \
  --train_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/training \
  --all_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/5par_all_phase.csv \
  --val_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/5par_supervised_valid.csv \
  --resume /home/fguo24/scratch/Reg_Seg/Pretrained_Reg/Checkpoint_Pretrained_Reg_series/trained_model.pth \
  --TPer_state_dict_resume /home/fguo24/scratch/Reg_Seg/TAtten_series/model/TPer.pth \
  --train_output_dir /home/fguo24/scratch/Reg_Seg/semi_2Seg_TAtten_features_5par_series \
  --test_output_dir /home/fguo24/scratch/Reg_Seg/semi_2Seg_TAtten_features_5par_series \
  --test_data_list /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/test_subj.csv \
  --test_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Time_Series_Seg/database/testing \
  --mode train \
  --epochs 250 \
  --learning_rate 4e-4