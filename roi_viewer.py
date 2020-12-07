import nibabel as nib
import glob
import numpy as np
import matplotlib.pylab as plt
import pydicom as dicom

file = nib.load(glob.glob('../ProstateMRF_forDL/PT12_120417_PROSTATEBX_LRH/*.nii.gz')[0])

img = file.get_fdata()

img = np.swapaxes(img, 0, 2)
print(img.shape)
plt.figure(figsize=(3, 3))
plt.gray()
plt.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)

for x, i in enumerate(img):
    image_path = glob.glob('../ProstateMRF_forDL/PT12_120417_PROSTATEBX_LRH/MRFimage{}.dcm'.format(x + 1))[0]
    im2 = dicom.dcmread(image_path)
    im2 = im2.pixel_array[6]
    mask = i > 0
    im2 = im2 * mask.astype(int)
    plt.subplot(3, 3, x + 1), plt.imshow(im2), plt.axis('off')
plt.show()