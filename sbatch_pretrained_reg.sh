#!/bin/bash
#SBATCH --account=def-gawright                  # using graham's account
#SBATCH --gpus-per-node=p100:1            	    # request a p100 GPU (v100 gpus does not work well for this code, not sure why)
#SBATCH --mem=90G                  				# memory per node
#SBATCH --time=00-23:00                         # reserve a gpu for 15 hours.
#SBATCH --output=DFmorph.out               # log output

source ../python3.7-pytorch1.5.1/bin/activate
python DFmorph.py \
    --train_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/yqin/ACDC/all/converted \
    --model_dir /home/fguo/scratch/Reg_Seg/Checkpoint_Pretrained_Reg \
    --log_dir  /home/fguo/scratch/Reg_Seg/Log_Pretrained_Reg \
    --result_dir /home/fguo/scratch/Reg_Seg/Result_Pretrained
