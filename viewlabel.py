import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import nibabel as nib
from scipy import interpolate

nib_labels = nib.load('patient139/patient139_ED_gt.nii.gz')
subj = nib.load('patient139/patient139_ED.nii.gz')
img = subj.get_fdata()

labels = nib_labels.get_fdata()
lab0 = np.ones_like(labels)
lab0[labels == 0] = 0
nib_lab0 = nib.Nifti1Image(lab0.astype('int16'), None, nib_labels.header)
nib.save(nib_lab0, 'patient139/lab0.nii.gz')

lab1 = np.zeros_like(labels)
lab1[labels == 1] = 1
nib_lab1 = nib.Nifti1Image(lab1.astype('int16'), None, nib_labels.header)
nib.save(nib_lab1, 'patient139/lab1.nii.gz')

lab2 = np.zeros_like(labels)
lab2[labels == 2] = 1
nib_lab2 = nib.Nifti1Image(lab2.astype('int16'), None, nib_labels.header)
nib.save(nib_lab2, 'patient139/lab2.nii.gz')

lab3 = np.zeros_like(labels)
lab3[labels == 3] = 1
nib_lab3 = nib.Nifti1Image(lab3.astype('int16'), None, nib_labels.header)
nib.save(nib_lab3, 'patient139/lab3.nii.gz')

x, y, z = labels.shape

img = img[x // 2 - 40:x // 2 + 40, y // 2 - 20:y // 2 + 60, 6]
lab = labels[x // 2 - 40:x // 2 + 40, y // 2 - 20:y // 2 + 60, 6]
lab0 = lab0[x // 2 - 40:x // 2 + 40, y // 2 - 20:y // 2 + 60, 6]
lab1 = lab1[x // 2 - 40:x // 2 + 40, y // 2 - 20:y // 2 + 60, 6]
lab2 = lab2[x // 2 - 40:x // 2 + 40, y // 2 - 20:y // 2 + 60, 6]
lab3 = lab3[x // 2 - 40:x // 2 + 40, y // 2 - 20:y // 2 + 60, 6]

fig = plt.figure(1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.tight_layout()
fig.set_size_inches(img.shape[1] / 80, img.shape[0] / 80)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig = plt.figure(2)
plt.imshow(lab, cmap='gray')
plt.axis('off')
plt.tight_layout()
fig.set_size_inches(img.shape[1] / 80, img.shape[0] / 80)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig = plt.figure(3)
plt.imshow(lab0, cmap='gray')
plt.axis('off')
plt.tight_layout()
fig.set_size_inches(img.shape[1] / 80, img.shape[0] / 80)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig = plt.figure(4)
plt.imshow(lab1, cmap='gray')
plt.axis('off')
plt.tight_layout()
fig.set_size_inches(img.shape[1] / 80, img.shape[0] / 80)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig = plt.figure(5)
plt.imshow(lab2, cmap='gray')
plt.axis('off')
plt.tight_layout()
fig.set_size_inches(img.shape[1] / 80, img.shape[0] / 80)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig = plt.figure(6)
plt.imshow(lab3, cmap='gray')
plt.axis('off')
plt.tight_layout()
fig.set_size_inches(img.shape[1] / 80, img.shape[0] / 80)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.show()
