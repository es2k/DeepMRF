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
import pandas as pd
from models.utils import dsc, gray2rgb, outline


bigdata = pd.read_excel('../ProstateMRF/ProstateMRF_forDL/prostateMRF_DL_forEdward.xlsx',
                        engine='openpyxl')
fns = glob.glob('../ProstateMRF/ProstateMRF_forDL/*100*/')
print(fns)
print(len(fns))
count = 0
bigarray = []
largestcolor=[]
bigt1t2=[]

mrfarr=[]
t1t2arr=[]
maskarr=[]

for folder in fns:
    files = glob.glob(folder + '/*.nii*')

    maxd = 0
    maxfn = None
    for file in files:
        try:
            img = nib.load(file).get_fdata()
        except:
            continue
        img = np.swapaxes(img, 0, 2)
        if img.shape[0] > maxd:
            maxd = img.shape[0]
            maxfn = file
    # print(maxfn, maxd)
    if maxd == 1:
        if len(glob.glob(folder + '/MRFimage*')) > 1:
            print("bad folder: " + folder)
            continue
    patientname = folder.split('ProstateMRF_forDL')[1][1:-1]
    if not os.path.exists('models/cnn/'):  # {}/'.format(patientname)):
        os.makedirs('models/cnn/')  # {}/'.format(patientname))
    # print(patientname)
    img = nib.load(maxfn).get_fdata()
    img = np.swapaxes(img, 0, 2)
    print(patientname)
    img = img.astype('uint8')

    t1ds = dicom.dcmread(folder + '/t1map.dcm')
    t1mat = t1ds.pixel_array
    t2ds = dicom.dcmread(folder + '/t2map.dcm')
    t2mat = t2ds.pixel_array

    number = patientname.split('PT')[1].split('_')[0]
    if int(number) > 147:
        continue
    df = pd.read_csv(folder + '/Allresults.csv')
    # print(df)

    if t1mat.shape == (400, 400):
        t1mat = np.array([t1mat])
        t2mat = np.array([t2mat])
    # print(t1mat.shape,t2mat.shape)
    for idx, slice in enumerate(img):
        num, labels = cv2.connectedComponents(slice)
        # print(num)
        regions = measure.regionprops(labels)
        if len(regions) == 0:
            continue
        #fig,ax=plt.subplots()



        ds = dicom.dcmread(folder + '/MRFimage{}.dcm'.format(idx + 1))

        rgbtif = ds.pixel_array[:3].astype('float')
        rgbtif = rgbtif.transpose(1, 2, 0)

        rgbtif-=np.amin(rgbtif,axis=(0,1))
        rgbtif/=np.amax(rgbtif,axis=(0,1))
        rgbtif = np.uint8(rgbtif * 255.999)

        t1t2 = np.dstack((t1mat[idx], t2mat[idx], np.zeros((400, 400))))
        #t1t2.transpose(2,0,1)
        t1t2 = np.uint8(t1t2 / np.amax(t1t2, axis=(0, 1)) * 255.999)

        gray = np.dot(rgbtif[..., :3], [0.2989, 0.5870, 0.1140])
        #ax.imshow(gray, cmap='gray')

        for i, props in enumerate(regions):
            # print(props.bbox)
            minr, minc, maxr, maxc = props.bbox
            dr, dc = maxr - minr, maxc - minc
            lside = max(dr, dc)

            r, c = np.rint(props.centroid)
            minr, maxr = int(r - lside), int(r + lside)
            minc, maxc = int(c - lside), int(c + lside)

            crop = slice[minr:maxr, minc:maxc]

            rgbcrop = rgbtif[ minr:maxr, minc:maxc, :]
            rgbcrop = cv2.resize(rgbcrop,(64,64))
            #print(rgbcrop.shape)
            t1t2crop = cv2.resize(t1t2[minr:maxr, minc:maxc, :], (64,64))
            #print(t1t2crop.shape)
            tmp = (slice[minr:maxr, minc:maxc] > 0).astype('float')
            maskcrop = cv2.resize(tmp, (64,64))
            #print(maskcrop.shape)


            '''print(rgbtif.mean(), np.amax(rgbtif, axis=(0, 1)))
            #im2 = Image.fromarray(rgbcrop, 'RGB').resize((64, 64))
            #im2.save('models/cnn/{}_rgb.tif'.format(count))

            im3 = Image.fromarray(t1t2[minr:maxr, minc:maxc, :], 'RGB').resize((64, 64))
            im3.save('models/cnn/{}_t1t2.tif'.format(count))

            bi = slice[minr:maxr, minc:maxc] > 0
            im4 = Image.fromarray(bi).resize((64, 64))
            im4.save('models/cnn/{}_mask.tif'.format(count))'''
            count+=1
            '''crop=crop>0
            im = Image.fromarray(crop).resize((64,64))
            im.save('models/cnn/{}_mask.tif'.format(count))

            rgbcrop = rgbtif[minr:maxr,minc:maxc,:]
            #print(rgbtif.mean(), np.amax(rgbtif, axis=(0, 1)))
            im2 = Image.fromarray(rgbcrop, 'RGB').resize((64,64))
            im2.save('models/cnn/{}_rgb.tif'.format(count))

            im3 = Image.fromarray(t1t2[minr:maxr,minc:maxc,:],'RGB').resize((64,64))
            im3.save('models/cnn/{}_t1t2.tif'.format(count))

            bi = slice[minr:maxr, minc:maxc]>0
            im4 = Image.fromarray(bi).resize((64,64))
            im4.save('models/cnn/{}_bi.tif'.format(count))

            count+=1'''
            break
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax.plot(bx, by, '-g', linewidth=1)
            break

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

images = ['models/cnn/0_rgb.tif','models/cnn/0_t1t2.tif']
mask = 'models/cnn/0_mask.tif'
mask = np.array(Image.open(mask))
for i in images:
    image=Image.open(i)
    image = np.array(image)
    print(image.shape, mask.min(),mask.max())
    image = outline(image, mask.astype('float'), color=[255, 255, 255])
    image = outline(image, mask.astype('float'), color=[255, 255, 255])
    plt.imshow(image)
    plt.show()
    '''filename = "{}-{}.png".format(p, str(s).zfill(2))
    filepath = os.path.join(args.predictions, filename)
    imsave(filepath, image)'''