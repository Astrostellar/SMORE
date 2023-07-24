import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict
import SimpleITK as sitk
import random
import numpy as np
from scipy import ndimage
import json
from utils import to_pixel_samples
import skimage.io as io
import cv2
from train_liif import save_nii
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def downsample(volume,k=4):

    sigma=((k-1)*0.5-1)*0.3+0.8
    volume_blur= ndimage.gaussian_filter(volume,(0,0,sigma))
    volume_down= ndimage.zoom(volume_blur,(1,1,1/k))
    

    return volume_down
    
def downsample1(volume,k=4):
    idxs=list(range(0,volume.shape[-1],k))
    volume_down=volume[:,:,idxs]
    
    return volume_down
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='syn_image_lr.png')
    parser.add_argument('--model', default="/hpc/data/home/bme/v-wangxin/szr_code/LIIF/sag_x4/_train_edsr-baseline-liif/epoch-last.pth")
    parser.add_argument('--resolution', default='1024,1024')
    parser.add_argument('--output', default='syn_image_hr.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    
    k=4
    nii_path="/hpc/data/home/bme/v-wangxin/skull_stripped/100206_3T_T1w_MPR1.nii.gz"
    Image=sitk.ReadImage(nii_path)      
    volume = sitk.GetArrayFromImage(Image)
    volume=(volume - volume.min()) / (volume.max() - volume.min())
    volume_down=downsample1(volume)
    spacing=Image.GetSpacing()
    save_nii(volume_down,(spacing[0]*k,spacing[1],spacing[2]),Image.GetOrigin(),Image.GetDirection(),'volume_down.nii.gz')
    volume_recon=np.zeros_like(volume)
    for i in tqdm(range(volume_down.shape[0])):
        img=volume_down[i]  # (320,64)
        #print(img.shape)
        img=cv2.resize(img,dsize=None,fx=k,fy=1,interpolation=cv2.INTER_LINEAR)  # (320,256)
        h, w = 320,256
        #print(img.shape)
        #exit(0)
        inp=torch.FloatTensor(img).view(1,1,*img.shape)
        coord = make_coord((h, w)).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        pred = batched_predict(model, inp.cuda(), coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
        pred = pred.clamp(0, 1).view(h, w).cpu().numpy()
        if i==161:
            cv2.imwrite('orig_{}.jpg'.format(i),img*255)
            cv2.imwrite('recon_{}.jpg'.format(i),pred*255)
        volume_recon[i]=pred
    save_nii(volume_recon,spacing,Image.GetOrigin(),Image.GetDirection(),'volume_recon_new.nii.gz')
    print("psnr: ",psnr(volume,volume_recon),"ssim: ",ssim(volume,volume_recon,multichannel=True))
        

