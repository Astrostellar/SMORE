import os
import time
import shutil
import math
import torch.nn as nn
import torch
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils.img_process_util import filter2D
from basicsr.utils import DiffJPEG, USMSharp
import random
import torch.nn.functional as F

jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts

def degradation(gt,opt,kernel1,kernel2,sinc_kernel):
        # gt=gt.cuda()
        # kernel1=kernel1.cuda()
        # kernel2=kernel2.cuda()
        # sinc_kernel=sinc_kernel.cuda()
        ori_h, ori_w = gt.size()[2:4]
        
     # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(gt, kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        
        # add noise
        gray_noise_prob = opt['gray_noise_prob']
        if np.random.uniform() < opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                    out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                    out,
                    scale_range=opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < opt['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out, size=(int(ori_h  * scale), int(ori_w * scale)), mode=mode)
        # add noise
        gray_noise_prob = opt['gray_noise_prob2']
        if np.random.uniform() <opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                    out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                    out,
                    scale_range=opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h , ori_w ), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h, ori_w ), mode=mode)
            out = filter2D(out,sinc_kernel)
        return out.squeeze().detach()


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])   # shape=(H*W,2)
    rgb = img.view(img.shape[0], -1).permute(1, 0) 
    return coord, rgb

def normalize(img):
    if img.max()-img.min()==0:
        print(img.min(),img.max())
    return (img - img.min()) / (img.max() - img.min())
    

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))
    
def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size,conv=default_conv, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)
        
class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=False):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=False):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


    
def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)
