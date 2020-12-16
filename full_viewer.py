import glob
import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np

t1_path = glob.glob('../ProstateMRF/*/*/t1map.dcm')
t2_path = glob.glob('../ProstateMRF/*/*/t2map.dcm')
for i in t2_path:
    ds=dicom.dcmread(i)
    print(i, ds.pixel_array.shape)
    plt.imshow(ds.pixel_array[0])
    plt.show()
'''plt.figure(figsize=(3,3))
plt.gray()
plt.subplots_adjust(0,0,1,1,0.01,0.01)
for x,i in enumerate(image_path):
    ds = dicom.dcmread(i)
    plt.subplot(3,3,x+1), plt.imshow(ds.pixel_array[6]), plt.axis('off')
plt.show()
'''
