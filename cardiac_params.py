import nibabel as nib
import numpy as np
import pandas as pd
import os

data_dir = os.getcwd()
print(data_dir)

name_list = pd.read_csv(os.path.join(data_dir, 'list_subj'), header=None)[0].to_list()
gt_list = ['{}_Pred.nii.gz'.format(i, i) for i in name_list]
pred_list = ['/home/fguo24/projects/def-ouriadov/fguo24/GuoLabStudents/yqin/' \
             'Time_Series_Seg/database/testing/patient144/{}_manu.nii.gz'.format(i, i) for i in name_list]

output_file = 'cardiac_params.csv'
# LV_param_pred = ['LV_EDV_pred', 'LV_ESV_pred', 'LV_EF_pred', 'LVM_ED_pred', 'LVM_ES_pred', 'LVSV_pred']
# RV_param_pred = ['RV_EDV_pred', 'RV_ESV_pred', 'RV_EF_pred', 'RVSV_pred']
#
# LV_param_gt = ['LV_EDV_gt', 'LV_ESV_gt', 'LV_EF_gt', 'LVM_ED_gt', 'LVM_ES_gt', 'LVSV_gt']
# RV_param_gt = ['RV_EDV_gt', 'RV_ESV_gt', 'RV_EF_gt', 'RVSV_gt']


df = pd.DataFrame(
    columns=['Subject', 'LV_volume_pred', 'LV_volume_gt', 'LVM_volume_pred', 'LVM_volume_gt', 'RV_volume_pred',
             'RV_volume_gt'])

for gt_name, pred_name in zip(gt_list, pred_list):
    print('Processing GT: {}, Prediction: {}'.format(gt_name, pred_name))

    nii_pred = nib.load(pred_name)
    nii_gt = nib.load(gt_name)

    pred = nii_pred.get_fdata()
    gt = nii_gt.get_fdata()

    v_1, v_2, v_3 = nii_pred.header['pixdim'][1:4]
    v_vox = v_1 * v_2 * v_3 / 1000.
    LV_vox_num_pred = np.nonzero(pred == 3)[0].shape[0]
    LV_volume_pred = LV_vox_num_pred * v_vox

    RV_vox_num_pred = np.nonzero(pred == 1)[0].shape[0]
    RV_volume_pred = RV_vox_num_pred * v_vox

    Myo_vox_num_pred = np.nonzero(pred == 2)[0].shape[0]
    Myo_volume_pred = Myo_vox_num_pred * v_vox

    LV_vox_num_gt = np.nonzero(gt == 3)[0].shape[0]
    LV_volume_gt = LV_vox_num_gt * v_vox

    RV_vox_num_gt = np.nonzero(gt == 1)[0].shape[0]
    RV_volume_gt = RV_vox_num_gt * v_vox

    Myo_vox_num_gt = np.nonzero(gt == 2)[0].shape[0]
    Myo_volume_gt = Myo_vox_num_gt * v_vox

    subject_slice_df = {'Subject': gt_name, 'LV_volume_pred': LV_volume_pred, 'LV_volume_gt': LV_volume_gt,
                        'LVM_volume_pred': Myo_volume_pred, 'LVM_volume_gt': Myo_volume_gt,
                        'RV_volume_pred': RV_volume_pred, 'RV_volume_gt': RV_volume_gt}
    df = df.append(subject_slice_df, ignore_index=True)

df = df.round(3)
df.to_csv(os.path.join(data_dir, output_file), index=False, na_rep='nan')

