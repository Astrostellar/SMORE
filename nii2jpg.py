import os
import pandas as pd
import numpy as np
from PIL import Image
import SimpleITK as sitk
import skimage.io as io
import shutil

def normalize(img):
    return (img-img.min())/(img.max()-img.min())
    
def center_crop(data, shape):
    if shape[0] <= data.shape[-2]:
        w_from = (data.shape[-2] - shape[0]) // 2
        w_to = w_from + shape[0]
        data = data[..., w_from:w_to, :]
    else:
        w_before = (shape[0] - data.shape[-2]) // 2
        w_after = shape[0] - data.shape[-2] - w_before
        pad = [(0, 0)] * data.ndim
        pad[-2] = (w_before, w_after)
        data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
    if shape[1] <= data.shape[-1]:
        h_from = (data.shape[-1] - shape[1]) // 2
        h_to = h_from + shape[1]
        data = data[..., :, h_from:h_to]
    else:
        h_before = (shape[1] - data.shape[-1]) // 2
        h_after = shape[1] - data.shape[-1] - h_before
        pad = [(0, 0)] * data.ndim
        pad[-1] = (h_before, h_after)
        data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
    return data    
    
def nii2jpg(nii_path,q,crop_size=256):
    assert q<0.5
    image = sitk.ReadImage(nii_path)
    img=sitk.GetArrayFromImage(image)
    print(img.dtype)
    length=img.shape[0]
    start = round(length * q) # inclusive
    stop = length - start # exclusive

    img=img[start:stop]
    for i in range(img.shape[0]):
        #im=normalize(img[i])
        im=center_crop(img[i],(crop_size,crop_size))
        io.imsave('{}/{}_{}.jpg'.format(save_dir,os.path.basename(nii_path)[:9],i+1),im)

if __name__ == '__main__':
    dir="/mnt/shared_storage/wangxin/ski10/image/"
    save_dir="/mnt/shared_storage/wangxin/ski10/sag_slices"
    train_dir="/mnt/shared_storage/wangxin/ski10/train"
    val_dir="/mnt/shared_storage/wangxin/ski10/val"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    #for nii in os.listdir(dir):
       #nii2jpg(nii_path=os.path.join(dir,nii),q=0.2)
       #exit(0)
    for f in os.listdir(save_dir):
        if int(f[6:9])>80:
            shutil.copy(os.path.join(save_dir,f),os.path.join(val_dir,f))
        else:
            shutil.copy(os.path.join(save_dir,f),os.path.join(train_dir,f))
            