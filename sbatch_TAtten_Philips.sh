#!/bin/bash
#SBATCH --account=def-gawright                  # using graham's account
#SBATCH --gpus-per-node=p100:1            	    # request a p100 GPU (v100 gpus does not work well for this code, not sure why)
#SBATCH --mem=90G                  				# memory per node
#SBATCH --time=00-23:00                         # reserve a gpu for 15 hours.
#SBATCH --output=TAtten_Philips.out               # log output

source ../../../python3.7-pytorch1.7/bin/activate
python pretrained_for_TAtten.py \
  --train_data_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/MMM_chall_1/MMM/Converted_Labeled \
  --train_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_training_Philips.csv \
  --train_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_training_Philips.csv \
  --valid_sup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/supervised_training_Philips.csv \
  --valid_unsup_data_csv /home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/Reg_Seg/pytorch/pytorch-deeplab_v3_plus/unsupervised_training_Philips.csv \
  --seed 42 \
  --train_output_dir /home/fguo24/scratch/Reg_Seg/TAtten_Philips \
  --batch_size 4 \
