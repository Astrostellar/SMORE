
import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR

import models
import utils
from test import eval_psnr
import SimpleITK as sitk
import random
import numpy as np
from scipy import ndimage
import json
from utils import to_pixel_samples
import skimage.io as io
import cv2

def crop_bg(image,contour = 20):
    
   
    threshold = 0.001
    foreground = image > threshold
    (x,) = np.nonzero(np.amax(foreground, axis=1))
    (y,) = np.nonzero(np.amax(foreground, axis=0))
    
    if x.size==0 or y.size==0:  ## bad cases that shouldn't occurred
        return 0,0,image.shape[0],image.shape[1]
    
    x_min=x.min() - contour if x.min() > contour else 0
    y_min=y.min() - contour if y.min() > contour else 0
   
    
    x_max=x.max()+contour if x.max()+contour<image.shape[0] else image.shape[0]
    y_max=y.max()+contour if y.max()+contour<image.shape[1] else image.shape[1]
   
                
    return x_min,y_min,x_max,y_max

def random_crop(data,shape):
    w,h=data.shape
    assert w>=shape[0] and h>=shape[1] 

    x_min,y_min,x_max,y_max=crop_bg(data,contour=w//2+5)
    assert x_max-x_min>shape[0] and y_max-y_min>shape[1] 

    w_start=random.randrange(x_min,x_max-shape[0])
    h_start=random.randrange(y_min,y_max-shape[1])
    
    return w_start,h_start

def down_sample(volume,k=4):
    idxs=list(range(0,volume.shape[-1],k))
    new_volume_list=[]
    for idx in idxs:
        if idx+k<volume.shape[-1]:
            slice=np.zeros_like(volume[:,:,0])
            for i in range(k):
                slice=slice+volume[:,:,i+idx]
            slice=slice/k
        else:
            slice=np.zeros_like(volume[:,:,0])
            for i in range(volume.shape[-1]-idx):
                slice=slice+volume[:,:,i+idx]
            slice=slice/(volume.shape[-1]-idx)
        new_volume_list.append(slice)
    new_volume=np.stack(new_volume_list,axis=-1)
    return new_volume

def save_nii(volumn,spacing,origin,direction,save_path):
    sr_img = sitk.GetImageFromArray(volumn)
    sr_img.SetSpacing(spacing)
    sr_img.SetOrigin(origin)
    sr_img.SetDirection(direction)
    sitk.WriteImage(sr_img,save_path)

class make_dataset(Dataset):
    def __init__(self, nii_path, patch_size=48, k=4, augment=True):
        Image=sitk.ReadImage(nii_path)      
        volume = sitk.GetArrayFromImage(Image)

        # strip the all-black slices
        foreground = volume > 0.001 
        (z,) = np.nonzero(np.amax(foreground, axis=(0,1)))   
        volume=volume[:,:,z.min():z.max()]
        # print(volume.shape)
        # simulate slices with larger thickness
        self.sigma=((k-1)*0.5-1)*0.3+0.8
        volume_blur= ndimage.gaussian_filter(volume,(0,0,self.sigma))
        # volume_down=down_sample(volume_blur,k)
        volume_down= ndimage.zoom(volume_blur,(1,1,1/k))
        # spacing=Image.GetSpacing()
        # save_nii(volume_down,(spacing[0]*k,spacing[1],spacing[2]),Image.GetOrigin(),Image.GetDirection(),'volume_down.nii.gz')
        # exit(0)
        
        # volume_blur= ndimage.gaussian_filter(volume,(0,0,1))
        # volume_down= ndimage.zoom(volume_blur,(1,1,1/k))

        volume_down= self.normalize(volume_down)
        self.lr_volume=volume_down

        self.k=k
        self.patch_size=patch_size
        self.augment=augment


    def __len__(self):
        return self.lr_volume.shape[-1]

    def normalize(self,img):
        if img.max()-img.min()==0:
            print(img.min(),img.max())
        return (img - img.min()) / (img.max() - img.min())

    def __getitem__(self, index):

        img=self.lr_volume[:,:,index]
        # simulate anisotropic voxel
        if random.sample([0,1],1)[0]:
            sigma=(0,self.sigma)
            fx=1
            fy=1/self.k
        else:
            sigma=(self.sigma,0)
            fx=1/self.k
            fy=1

       
        img_blur= ndimage.gaussian_filter(img,sigma)
        # img_blur=cv2.GaussianBlur(img,ksize=self.k,sigmaX=self.sigma)

        img_down=cv2.resize(img_blur,dsize=None,fx=fx,fy=fy,interpolation=cv2.INTER_LINEAR)
        img_up=cv2.resize(img_down,dsize=None,fx=1/fx,fy=1/fy,interpolation=cv2.INTER_LINEAR)
        # img_down= ndimage.zoom(img_blur,(1/zoom[0],1/zoom[1]))
        # img_up= ndimage.zoom(img_down,zoom)
        # io.imsave('samples/img_up_{}.jpg'.format(index),(img_up*255).astype(np.uint8))
        assert img.shape==img_up.shape
        # random crop
        w_start,h_start=random_crop(img,(self.patch_size,self.patch_size))
        crop_hr=img[w_start:w_start+self.patch_size,h_start:h_start+self.patch_size]
        crop_lr=img_up[w_start:w_start+self.patch_size,h_start:h_start+self.patch_size]

        crop_hr=torch.FloatTensor(crop_hr).unsqueeze(0)
        crop_lr=torch.FloatTensor(crop_lr).unsqueeze(0)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    # dataset = datasets.make(spec['dataset'])
    # dataset = datasets.make(spec['wrapper'], args={'dataset': dataset,'opt':config.get('opt')})
    data="/hpc/data/home/bme/v-wangxin/new_ep/new_skull_stripped.json"
    with open(data,'r') as f:
            f_dict=json.load(fp=f)
    f.close()
    # dataset=make_dataset(f_dict[tag][0])

    dataset = torch.utils.data.ConcatDataset([make_dataset(nii_path) for nii_path in f_dict[tag]])
    
    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()


    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        pred = model(batch['inp'], batch['coord'], batch['cell'])

        loss = loss_fn(pred, batch['gt'])
        # print(pred.shape,gt.shape)
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None

    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./sag_x4', save_name)

    main(config, save_path)
