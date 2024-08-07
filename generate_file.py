import csv
import numpy as np
import random

# phases = ['ED', 'ES']
# subj = []
# with open('Philips.csv', mode='r', encoding='utf-8') as f:
#     reader = csv.reader(f)
#     for line in reader:
#         subj.append(line[0])
# indices = [i for i in range(125)]
# random.shuffle(indices)
# partial_indices = indices[:50]
# partial_indices_after = indices[50:]
# # with open('10par_supervised_training_MMM.csv', mode='w', newline='', encoding='utf-8') as f:
# #     writer = csv.DictWriter(f, fieldnames=['image_filenames', 'label_filenames'])
# #     writer.writeheader()
# #     for idx in partial_indices:
# #         prob = np.random.randint(0, 2)
# #         writer.writerow(
# #             {'image_filenames': '{}_{}/Img_resamp_12.nii.gz'.format(subj[int(idx)], phases[prob]),
# #              'label_filenames': '{}_{}/GT_resamp_12.nii.gz'.format(subj[int(idx)], phases[prob])}
# #         )
# with open('supervised_training_Philips.csv', mode='w', newline='', encoding='utf-8') as f:
#     writer = csv.DictWriter(f, fieldnames=['image_filenames', 'label_filenames'])
#     writer.writeheader()
#
#     for idx in partial_indices_after:
#         p = np.random.randint(0, 2)
#         writer.writerow(
#             {'image_filenames': '{}_{}/Img_resamp_12.nii.gz'.format(subj[int(idx)], phases[p]),
#              'label_filenames': '{}_{}/GT_resamp_12.nii.gz'.format(subj[int(idx)], phases[p])}
#         )
# with open('20par_supervised_training_Philips.csv', mode='w', newline='', encoding='utf-8') as f:
#     writer = csv.DictWriter(f, fieldnames=['image_filenames', 'label_filenames'])
#     writer.writeheader()
#
#     for idx in partial_indices_after[:int(0.4 * len(partial_indices_after))]:
#         p = np.random.randint(0, 2)
#         writer.writerow(
#             {'image_filenames': '{}_{}/Img_resamp_12.nii.gz'.format(subj[int(idx)], phases[p]),
#              'label_filenames': '{}_{}/GT_resamp_12.nii.gz'.format(subj[int(idx)], phases[p])}
#         )
# with open('10par_supervised_training_Philips.csv', mode='w', newline='', encoding='utf-8') as f:
#     writer = csv.DictWriter(f, fieldnames=['image_filenames', 'label_filenames'])
#     writer.writeheader()
#
#     for idx in partial_indices_after[:int(0.2 * len(partial_indices_after))]:
#         p = np.random.randint(0, 2)
#         writer.writerow(
#             {'image_filenames': '{}_{}/Img_resamp_12.nii.gz'.format(subj[int(idx)], phases[p]),
#              'label_filenames': '{}_{}/GT_resamp_12.nii.gz'.format(subj[int(idx)], phases[p])}
#         )
# with open('test_subj_Philips.csv', mode='w', newline='', encoding='utf-8') as f:
#     writer = csv.DictWriter(f, fieldnames=['image_filenames', 'label_filenames'])
#     writer.writeheader()
#
#     for idx in partial_indices:
#         writer.writerow(
#             {'image_filenames': '{}_ED/Img_resamp_12.nii.gz'.format(subj[int(idx)]),
#              'label_filenames': '{}_ED/GT_resamp_12.nii.gz'.format(subj[int(idx)])}
#         )
#         writer.writerow(
#             {'image_filenames': '{}_ES/Img_resamp_12.nii.gz'.format(subj[int(idx)]),
#              'label_filenames': '{}_ES/GT_resamp_12.nii.gz'.format(subj[int(idx)])}
#         )
subj = []
with open('10par_supervised_training_Philips.csv', mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for line in reader:
        subj.append(line['image_filenames'].split('_')[0])
with open('10par_all_training_Philips.csv', mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['image_filenames', 'label_filenames'])
    writer.writeheader()

    for sbj in subj:
        writer.writerow(
            {'image_filenames': '{}_ED/Img_resamp_12.nii.gz'.format(sbj),
             'label_filenames': '{}_ED/GT_resamp_12.nii.gz'.format(sbj)}
        )
        writer.writerow(
            {'image_filenames': '{}_ES/Img_resamp_12.nii.gz'.format(sbj),
             'label_filenames': '{}_ES/GT_resamp_12.nii.gz'.format(sbj)}
        )