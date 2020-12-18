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

# file = nib.load(glob.glob('../ProstateMRF_forDL/PT12_120417_PROSTATEBX_LRH/*.nii.gz')[0])

bigdata = pd.read_excel('../ProstateMRF/ProstateMRF_forDL/prostateMRF_DL_forEdward.xlsx',
                        engine='openpyxl')
fns = glob.glob('../ProstateMRF/ProstateMRF_forDL/*/')
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
        # fig,ax=plt.subplots()
        # ax.imshow(slice)

        ds = dicom.dcmread(folder + '/MRFimage{}.dcm'.format(idx + 1))
        rgbtif = ds.pixel_array[:3] #.astype('float')
        rgbtif = rgbtif.transpose(1, 2, 0)

        '''for color in range(3):
            mean, std = cv2.meanStdDev(rgbtif[color])
            print(mean,std)
            #result = (rgbtif-mean)/std
            #offset=10
            #rgbtif[color] = np.clip(rgbtif[color], mean - offset * std, mean + offset * std) #.astype(np.uint8)
            rgbtif[color] = cv2.normalize(rgbtif[color], None, 0, 1, norm_type=cv2.NORM_MINMAX)'''

        #print(result.min(),result.max(),result.mean(),result.std())
        '''largestcolor.append(np.amax(rgbtif,axis=(0,1)))
        rgbtif-=np.amin(rgbtif,axis=(0,1))
        rgbtif/=np.amax(rgbtif,axis=(0,1))
        rgbtif = np.uint8(rgbtif * 255.999)'''

        # rgbtif/rgbtif.max()
        #rgbtif = rgbtif.transpose(1, 2, 0)


        t1t2 = np.dstack((t1mat[idx], t2mat[idx], np.zeros((400, 400))))
        #t1t2.transpose(2,0,1)
        '''bigt1t2.append(np.amax(t1t2,axis=(0,1)))
        t1t2 = np.uint8(t1t2 / np.amax(t1t2, axis=(0, 1)) * 255.999)'''

        for i, props in enumerate(regions):
            # print(props.bbox)
            minr, minc, maxr, maxc = props.bbox
            dr, dc = maxr - minr, maxc - minc
            lside = max(dr, dc)

            r, c = np.rint(props.centroid)
            minr, maxr = int(r - lside), int(r + lside)
            minc, maxc = int(c - lside), int(c + lside)
            # print(r,c)
            crop = slice[minr:maxr, minc:maxc]
            a, b = np.unique(crop, return_counts=True)
            print(patientname, idx, a,b)
            where = np.argmax(b[1:])
            label = a[1:][where]
            row = df.loc[df['Label Id'] == label]
            print(label)
            name = row['Label Name'].iloc[0]

            if "Lesion" not in name:
                bigarray.append('0')
            else:
                lol = name.split('Lesion ')[1]
                print(number, lol)

                row = bigdata.loc[(bigdata['MRN'] == int(number)) &
                                  (bigdata['Lesion_number'] == int(lol))]
                pirads = row['PIRADS'].iloc[0]
                print(pirads)
                bigarray.append(str(int(pirads)))

            rgbcrop = rgbtif[ minr:maxr, minc:maxc, :]
            rgbcrop = cv2.resize(rgbcrop,(64,64))
            #print(rgbcrop.shape)
            t1t2crop = cv2.resize(t1t2[minr:maxr, minc:maxc, :], (64,64))
            #print(t1t2crop.shape)
            tmp = (slice[minr:maxr, minc:maxc] > 0).astype('float')
            maskcrop = cv2.resize(tmp, (64,64))
            #print(maskcrop.shape)
            mrfarr.append(rgbcrop)
            t1t2arr.append(t1t2crop)
            maskarr.append(maskcrop)

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

            # bx = (minc, maxc, maxc, minc, minc)
            # by = (minr, minr, maxr, maxr, minr)
            # ax.plot(bx, by, '-b', linewidth=1)
        # print()
        # plt.show()
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

print(bigarray)
np.savetxt('models/cnn/labels.txt',bigarray,fmt='%s')
mrfarr = np.array(mrfarr).transpose(0,3,1,2)
t1t2arr = np.array(t1t2arr).transpose(0,3,1,2)
maskarr = np.array(maskarr)
print(mrfarr.shape,t1t2arr.shape,maskarr.shape)
with open('models/alldata.npy','wb') as f:
    np.save(f, [mrfarr,t1t2arr])
