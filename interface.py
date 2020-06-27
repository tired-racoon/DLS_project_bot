import sys
import os

import net
import gan
import utils

import glob

import torch

def create_net():
    root = os.path.dirname(__file__)

    n = net.Net()

    msgnet_checkpoint = os.path.join(root, 'msgnet.pth')
    if os.path.exists(msgnet_checkpoint):
        n.load_state_dict(torch.load(msgnet_checkpoint))
    n.eval()
    return n

def create_gan():
    root = os.path.dirname(__file__)

    n = gan.Generator(4, 4)

    gun_weights = os.path.join(root, 'gan.pth')
    if os.path.exists(gun_weights):
        n.load_state_dict(torch.load(gun_weights))
    n.eval()
    return n

def gan_to_cpu():
    root = os.path.dirname(__file__)
    gun_weights = os.path.join(root, 'gan.pth')
    n = create_gan().cpu()
    n.eval()
    torch.save(n.state_dict(), gun_weights)

def style_images():
    root = os.path.dirname(__file__)
    orig_styles = os.path.join(root, 'dataset_style', '*.jpg')
    orig_styles = sorted(glob.glob(orig_styles))
    return orig_styles

def examples_images():
    root = os.path.dirname(__file__)
    imgs_root = os.path.join(root, 'examples', 'images')
    styles = style_images()
    imgs = list(map(lambda path: os.path.join(imgs_root, os.path.split(path)[-1]), styles))
    return imgs

def examples_styles_images():
    root = os.path.dirname(__file__)
    imgs_root = os.path.join(root, 'examples', 'images')
    styles = style_images()
    imgs = list(map(lambda path: os.path.join(imgs_root, os.path.split(path)[-1]), styles))
    return zip(styles, imgs)

def example_content():
    root = os.path.dirname(__file__)
    return os.path.join(root, 'examples', 'ico.jpg')

def do_style(msgnet, style, content, image):
    with torch.no_grad():
        if type(content) is str:
            content = utils.tensor_load_rgbimage(content, 256, keep_asp=True)
        msgnet.setTarget(utils.tensor_load_rgbimage(style, 256).unsqueeze(0))
        im = msgnet(content.detach().unsqueeze(0))[0]
        utils.tensor_save_rgbimage(im, image)

def zero_embeddings(batch):
    return torch.zeros((batch.shape[0], 1, batch.shape[2], batch.shape[3]), device=batch.device)
def cat_embeddings(batch, embeddings):
    return torch.cat((batch, embeddings), dim=1)
def split_only_batch(batch):
    return batch[:,0:-1,:,:]
def gan_normalize(t):
    t = t / 256.
    t[0] = (t[0] - 0.5) / 0.5
    t[1] = (t[1] - 0.5) / 0.5
    t[2] = (t[2] - 0.5) / 0.5
    return t
def gan_unnormalize(t):
    t = t * 1.
    t[0] = t[0] * 0.5 + 0.5
    t[1] = t[1] * 0.5 + 0.5
    t[2] = t[2] * 0.5 + 0.5
    t = t * 256.
    return t
def do_gan(generator, content, image):
    with torch.no_grad():
        if type(content) is str:
            content = utils.tensor_load_rgbimage(content, 256, keep_asp=False, need_normalize=False)
        content = gan_normalize(content)
        content = content.unsqueeze(0)
        content = cat_embeddings(content, zero_embeddings(content))
        im = split_only_batch(generator(content))[0]
        im = gan_unnormalize(im)
        utils.tensor_save_rgbimage(im, image, need_unnormalize=False)



def make_examples():
    ico = utils.tensor_load_rgbimage(example_content(), keep_asp=True)

    msgnet = create_net()
    for s, i in examples_styles_images():
        do_style(msgnet, s, ico, i)

if __name__ == '__main__':
    pass
    #make_examples()
    #g = create_gan()
    #do_gan(g, 'examples/ico.jpg', 'examples/g.jpg')
    #gan_to_cpu()
