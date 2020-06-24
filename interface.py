import sys
import os

import net
import utils

import glob

import torch

def create_net():
    root = os.path.dirname(__file__)

    n = net.Net()

    msgnet_checkpoint = os.path.join(root, 'msgnet.pth')
    if os.path.exists(msgnet_checkpoint):
        n.load_state_dict(torch.load(msgnet_checkpoint))

    n = n.cpu()
    n.eval()
    return n

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
    if type(content) is str:
        content = utils.tensor_load_rgbimage(content, 256, keep_asp=True)
    msgnet.setTarget(utils.tensor_load_rgbimage(style, 256).unsqueeze(0))
    with torch.no_grad():
        im = msgnet(content.unsqueeze(0))[0]
    utils.tensor_save_rgbimage(im, image)


def make_examples():
    ico = utils.tensor_load_rgbimage(example_content(), keep_asp=True)

    msgnet = create_net()
    for s, i in examples_styles_images():
        do_style(msgnet, s, ico, i)

if __name__ == '__main__':
    make_examples()