import argparse
import types

__all__=[
    'test',
    'dict2namespace'
]


import torchvision.transforms as transforms
from PIL import Image
import torch
import os

def imgdir_to_tensor(content_dir,content_shape,order="CHW",max_num=30,extensions=['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']):
    '''
    Read a image or images from a dir then convert them to tensor.

    Args:
        content_dir(str): image path.
        content_shape:[int,int]: reshape all images. [H,W] [-1,-1] means no reshape

        order: CHW or HWC

        extensions: images to read. 
        max_num: max images to return default 30

    Return:
        Return a image tensor  [N,C,H,W]

    '''

    ts=None
    if content_shape[0] <=0:
        ts= transforms.Compose([
        # transforms.Resize(content_shape),
        transforms.ToTensor() #[0,255] ->>[0,1]
        ])
    else:
        ts= transforms.Compose([
        transforms.Resize(content_shape),
        transforms.ToTensor() #[0,255] ->>[0,1]
        ])

    im_list = []

    if os.path.isfile(content_dir):
        # single file
        content_image = Image.open(content_dir)
        im_list.append(ts(content_image)) # after transform is [C,H,W]
    elif os.path.isdir(content_dir):
        # dir files
        for file_name in os.listdir(content_dir):
            if any(file_name.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']):
                content_image = Image.open(os.path.join(content_dir,file_name))
                im_list.append(ts(content_image)) # after transform is [C,H,W] 
    else:
        raise ValueError("Dot a dir or file")

    if (len(im_list) > max_num ):
        im_list = im_list[0:max_num]

    img_tensor = torch.stack(im_list,dim=0) # [N,C,H,W]

    if order =='HWC':
        img_tensor = img_tensor.permute(0,2,3,1)

    return img_tensor

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

# 递归函数将Namespace对象转换为字典
def namespace2dict(namespace_obj):
    if isinstance(namespace_obj, argparse.Namespace):
        return {key: namespace2dict(value) for key, value in vars(namespace_obj).items()}
    else:
        return namespace_obj


