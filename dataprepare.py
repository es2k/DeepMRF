import nibabel as nib
import glob
import numpy as np
from PIL import Image
import os
import pydicom as dicom
import matplotlib.pylab as plt
import pydicom as dicom
from skimage import measure
import cv2

#file = nib.load(glob.glob('../ProstateMRF_forDL/PT12_120417_PROSTATEBX_LRH/*.nii.gz')[0])

fns = glob.glob('../ProstateMRF/ProstateMRF_forDL/*/')
print(fns)
print(len(fns))
count=0
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
    if not os.path.exists('models/crop/'): #{}/'.format(patientname)):
        os.makedirs('models/crop/') #{}/'.format(patientname))
    #print(patientname)
    img = nib.load(maxfn).get_fdata()
    img = np.swapaxes(img, 0, 2)
    print(patientname)
    img=img.astype('uint8')

    t1ds = dicom.dcmread(folder+'/t1map.dcm')
    t1mat = t1ds.pixel_array
    t2ds = dicom.dcmread(folder+'/t2map.dcm')
    t2mat = t2ds.pixel_array
    if t1mat.shape == (400,400):
        t1mat = np.array([t1mat])
        t2mat = np.array([t2mat])
    print(t1mat.shape,t2mat.shape)
    for idx, slice in enumerate(img):
        num,labels = cv2.connectedComponents(slice)
        #print(num)
        regions = measure.regionprops(labels)
        if len(regions)==0:
            continue
        #fig,ax=plt.subplots()
        #ax.imshow(slice)

        ds = dicom.dcmread(folder + '/MRFimage{}.dcm'.format(idx + 1))
        rgbtif = ds.pixel_array[:3]
        # rgbtif/rgbtif.max()
        rgbtif = rgbtif.transpose(1, 2, 0)
        rgbtif = np.uint8(rgbtif / np.amax(rgbtif, axis=(0, 1)) * 255.999)

        t1t2 = np.dstack((t1mat[idx], t2mat[idx], np.zeros((400,400))))
        t1t2 = np.uint8(t1t2 / np.amax(t1t2, axis=(0, 1)) * 255.999)

        for i, props in enumerate(regions):
            #print(props.bbox)
            minr, minc, maxr, maxc = props.bbox
            dr,dc = maxr-minr,maxc-minc
            lside = max(dr,dc)
            r,c = np.rint(props.centroid)
            minr,maxr = int(r-lside), int(r+lside)
            minc,maxc = int(c-lside), int(c+lside)
            #print(r,c)
            crop=slice[minr:maxr,minc:maxc].astype('float')
            im = Image.fromarray(crop).resize((256,256))
            im.save('models/crop/{}_mask.tif'.format(count))

            rgbcrop = rgbtif[minr:maxr,minc:maxc,:]
            #print(rgbtif.mean(), np.amax(rgbtif, axis=(0, 1)))
            im2 = Image.fromarray(rgbcrop, 'RGB').resize((256,256))
            im2.save('models/crop/{}_rgb.tif'.format(count))

            im3 = Image.fromarray(t1t2[minr:maxr,minc:maxc,:],'RGB').resize((256,256))
            im3.save('models/crop/{}_t1t2.tif'.format(count))

            bi = slice[minr:maxr, minc:maxc]>0
            im4 = Image.fromarray(bi).resize((256,256))
            im4.save('models/crop/{}_bi.tif'.format(count))

            count+=1
            #bx = (minc, maxc, maxc, minc, minc)
            #by = (minr, minr, maxr, maxr, minr)
            #ax.plot(bx, by, '-b', linewidth=1)
        #print()
        #plt.show()
        '''
        slice=slice>0
        im = Image.fromarray(slice)
        im.save('models/crop/{}/{}_mask.tif'.format(patientname, idx))
        ds = dicom.dcmread(folder+'/MRFimage{}.dcm'.format(idx+1))
        rgbtif = ds.pixel_array[:3]
        #rgbtif/rgbtif.max()
        rgbtif = rgbtif.transpose(1,2,0)
        rgbtif = np.uint8(rgbtif/np.amax(rgbtif,axis=(0,1))*255.999)
        print(rgbtif.mean(),np.amax(rgbtif, axis=(0,1)))
        im2=Image.fromarray(rgbtif,'RGB')
        im2.save('models/crop/{}/{}_rgb.tif'.format(patientname,idx))'''
print(count)