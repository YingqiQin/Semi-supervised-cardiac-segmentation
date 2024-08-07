import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import nibabel as nib
from scipy import interpolate

for i in 'UAMT', 'DAN', 'URPC', 'CCT', 'CPS', 'DeepAtlas', 'BRBS', 'Ours':
    subj = nib.load('patient139/patient139_ES.nii.gz')
    subj_gt = nib.load(f'{i}/10par/patient139/patient139_ES_Pred.nii.gz')
    img = subj.get_fdata()
    seg = subj_gt.get_fdata()
    x, y, z = img.shape
    img = img[x // 2 - 40:x // 2 + 40, y // 2 - 20:y // 2 + 60, :]
    seg = seg[x // 2 - 40:x // 2 + 40, y // 2 - 20:y // 2 + 60, :]
    slice_num = 6

    img = np.rot90(np.fliplr(img[:, :, slice_num]))
    seg = np.rot90(np.fliplr(seg[:, :, slice_num]))

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    lv_endo = (seg == 1).astype(np.uint8)

    if np.sum(lv_endo) > 0:
        contours = skimage.measure.find_contours(lv_endo, 0.5)

        for _, contour_pieces in enumerate(contours):
            spline_order = min(3, len(contour_pieces[:, 0]) - 1)
            tck, u = interpolate.splprep([contour_pieces[:, 1], contour_pieces[:, 0]], s=1, k=spline_order)
            unew = np.arange(0, 1.01, 0.01)
            out = interpolate.splev(unew, tck)
            plt.plot(out[0], out[1], color='cyan', linewidth=1)

    whole_part = (seg == 2).astype(np.uint8)
    if np.sum(whole_part) > 0:
        contours = skimage.measure.find_contours(whole_part, 0.5)

        for _, contour_pieces in enumerate(contours):
            spline_order = min(3, len(contour_pieces[:, 0]) - 1)
            tck, u = interpolate.splprep([contour_pieces[:, 1], contour_pieces[:, 0]], s=1, k=spline_order)
            unew = np.arange(0, 1.01, 0.01)
            out = interpolate.splev(unew, tck)
            plt.plot(out[0], out[1], color='orange', linewidth=1)

    rv = (seg == 3).astype(np.uint8)

    if np.sum(rv) > 0:
        contours = skimage.measure.find_contours(rv, 0.5)

        for _, contour_pieces in enumerate(contours):
            spline_order = min(3, len(contour_pieces[:, 0]) - 1)
            tck, u = interpolate.splprep([contour_pieces[:, 1], contour_pieces[:, 0]], s=1, k=spline_order)
            unew = np.arange(0, 1.01, 0.01)
            out = interpolate.splev(unew, tck)
            plt.plot(out[0], out[1], color='red', linewidth=1)
    plt.axis('off')
    plt.tight_layout()
    fig.set_size_inches(img.shape[1] / 80, img.shape[0] / 80)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(f'pictures/{i}_10.png', dpi=900, bbox_inches='tight', pad_inches=0)
    plt.show()
