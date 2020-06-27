import os

import numpy as np
import torch
from PIL import Image

def normalize(t):
    t = t / 256.
    t[0] = (t[0] - 0.485) / 0.229
    t[1] = (t[1] - 0.456) / 0.224
    t[2] = (t[2] - 0.406) / 0.225
    return t

def unnormalize(t):
    t = t * 1.
    t[0] = t[0] * 0.229 + 0.485
    t[1] = t[1] * 0.224 + 0.456
    t[2] = t[2] * 0.225 + 0.406
    t = t * 256.
    return t

def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False, need_normalize=True):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    if need_normalize:
        img = normalize(img)
    return img


def tensor_save_rgbimage(tensor, filename, cuda=False, need_unnormalize=True):
    if need_unnormalize:
        tensor = unnormalize(tensor)
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

class StyleLoader():
    def __init__(self, style_folder, style_size, cuda=True):
        self.folder = style_folder
        self.style_size = style_size
        self.files = os.listdir(style_folder)
        self.cuda = cuda
    
    def get(self, i):
        idx = i%len(self.files)
        filepath = os.path.join(self.folder, self.files[idx])
        style = tensor_load_rgbimage(filepath, self.style_size)
        style = style.unsqueeze(0)
        if self.cuda:
            style = style.cuda()
        return style

    def size(self):
        return len(self.files)