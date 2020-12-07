import glob
import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np

image_path = glob.glob('../ProstateMRF_forDL/PT12_120417_PROSTATEBX_LRH/MRF*.dcm')
print(image_path)

plt.figure(figsize=(3,3))
plt.gray()
plt.subplots_adjust(0,0,1,1,0.01,0.01)
for x,i in enumerate(image_path):
    ds = dicom.dcmread(i)
    plt.subplot(3,3,x+1), plt.imshow(ds.pixel_array[6]), plt.axis('off')
plt.show()

