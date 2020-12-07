import nibabel as nib
import glob
import numpy as np
from PIL import Image
import os
import pydicom as dicom
import matplotlib.pylab as plt
import pydicom as dicom

#file = nib.load(glob.glob('../ProstateMRF_forDL/PT12_120417_PROSTATEBX_LRH/*.nii.gz')[0])

fns = glob.glob('../ProstateMRF/ProstateMRF_forDL/*/')
print(fns)
print(len(fns))
c=0
for folder in fns:
    files = glob.glob(folder + '/*.nii*')

    maxd = 0
    maxfn=None
    for file in files:
        try:
            img = nib.load(file).get_fdata()
        except:
            continue
        img = np.swapaxes(img, 0, 2)
        if img.shape[0]>maxd:
            maxd=img.shape[0]
            maxfn = file
    #print(maxfn, maxd)
    if maxd==1:
        if len(glob.glob(folder+'/MRFimage*'))>1:
            print("bad folder: " + folder)
            continue
    patientname = folder.split('ProstateMRF_forDL')[1][1:-1]
    if not os.path.exists('models/intermediate/{}/'.format(patientname)):
        os.makedirs('models/intermediate/{}/'.format(patientname))
    #print(patientname)
    img = nib.load(maxfn).get_fdata()
    img = np.swapaxes(img, 0, 2)
    for idx, slice in enumerate(img):
        slice=slice>0
        im = Image.fromarray(slice)
        im.save('models/intermediate/{}/{}_mask.tif'.format(patientname, idx))
        ds = dicom.dcmread(folder+'/MRFimage{}.dcm'.format(idx+1))
        rgbtif = ds.pixel_array[:3]
        #rgbtif/rgbtif.max()
        rgbtif = rgbtif.transpose(1,2,0)
        rgbtif = np.uint8(rgbtif/np.amax(rgbtif,axis=(0,1))*255.999)
        print(rgbtif.mean(),np.amax(rgbtif, axis=(0,1)))
        im2=Image.fromarray(rgbtif,'RGB')
        im2.save('models/intermediate/{}/{}_rgb.tif'.format(patientname,idx))
