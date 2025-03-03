# ref https://github.com/Xian-Bei/CLUT & https://github.com/semchan/NLUT

import numpy as np
import math
import torch
import os

# Ref https://github.com/semchan/NLUT/blob/main/utils/LUT.py
def identity3d_tensor(dim): # 3,d,d,d #[rgb,b,g,r]
    '''
        Generate identity LUT in tensor shape [3,dim,dim,dim] rep [rgb color ,b dim g dim, r dim]
    '''
    
    step = np.arange(0,dim)/(dim-1) # Double, so need to specify dtype # shape:(dim,)
    rgb = torch.tensor(step, dtype=torch.float32)# np to tensor (dim)
    LUT = torch.empty(3,dim,dim,dim)# tensor 
    LUT[0] = rgb.unsqueeze(0).unsqueeze(0).expand(dim, dim, dim) # r
    LUT[1] = rgb.unsqueeze(-1).unsqueeze(0).expand(dim, dim, dim) # g
    LUT[2] = rgb.unsqueeze(-1).unsqueeze(-1).expand(dim, dim, dim) # b
    return LUT

def identity2d_tensor(dim): # 2,d,d
    # Double, so need to specify dtype
    step = torch.tensor(np.arange(0,dim)/(dim-1), dtype=torch.float32)
    hs = torch.empty(2,dim,dim)
    hs[0] = step.unsqueeze(0).repeat(dim, 1) # r # (dim)>(1,dim)>(dim,dim)
    hs[1] = step.unsqueeze(1).repeat(1, dim) # g
    return hs
    
def identity1d_tensor(dim): # 1,d
    step = np.arange(0,dim)/(dim-1) # Double, so need to specify dtype
    return torch.tensor(step, dtype=torch.float32).unsqueeze(0)

def read_3dlut_from_file(file_name, return_type="tensor"):
    """
    Read 3dLut from given file.

    Args:
        file_name(str): file location.
        return_type: "tensor" or "np"

    Returns:
        LUT tensor [3,dim B, dim G, dim R]

    """

    file = open(file_name, 'r')
    lines = file.readlines()
    start, end = 0, 0 # 从cube文件读取时
    for i in range(len(lines)):
        if lines[i][0].isdigit() or lines[i].startswith("-"):
            start = i
            break
    for i in range(len(lines)-1,start,-1):
        if lines[i][0].isdigit() or lines[i].startswith("-"):
            end = i
            break
    lines = lines[start: end+1]
    if len(lines) == 262144:
        dim = 64
    elif len(lines) == 35937:
        dim = 33
    else:
        dim = int(np.round(math.pow(len(lines), 1/3)))
    print("dim = ", dim)
    buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)
    # LUT的格式是 cbgr，其中c是 rgb
    # 在lut文件中，一行中依次是rgb
    # r是最先最多变化的，b是变化最少的
    # 往里填的过程中，k是最先最多变化的，它填在最后位置
    for i in range(0,dim):# b
        for j in range(0,dim):# g
            for k in range(0,dim):# r
                n = i * dim*dim + j * dim + k
                x = lines[n].split()
                buffer[0,i,j,k] = float(x[0])# r
                buffer[1,i,j,k] = float(x[1])# g
                buffer[2,i,j,k] = float(x[2])# b

    if return_type in["numpy", "np"]:
        return buffer
    elif return_type in["tensor", "ts"]:
        return torch.from_numpy(buffer)
        # buffer = torch.zeros(3,dim,dim,dim) # 直接用torch太慢了，不如先读入np再直接转torch
    else:
        raise ValueError("return_type should be np or ts")
    
def save_3dlut_to_file(lut, file_path,name="Test"):
    '''
        Save a lut to file

        Args:
            lut (tensor): 3D lut tensor [3,dim,dim,dim]
            file_path : path to save.
            name: file name ends with .cube
    '''
    assert lut.shape[2] == lut.shape[1]
    assert lut.shape[2] == lut.shape[3]

    dim = lut.shape[1]
    if not os.path.exists(file_path):
        os.makedirs(file_path,exist_ok=True)

    path = os.path.join(file_path,name+".CUBE")

    with open(path, 'w') as f:
        f.write(f'TITLE "{name}" \n \n')
        f.write("#LUT size \n")
        f.write(f"LUT_3D_SIZE {dim} \n\n")
        f.write('DOMAIN_MIN 0.0 0.0 0.0 \n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0 \n\n')
        f.write('#data point \n')

        for i in range(lut.shape[1]):#b
            for j in range(lut.shape[2]):#g
                for k in range(lut.shape[3]): #r
                    f.write('{:.6f}  {:.6f}  {:.6f}\n'.format(
                        lut[0, i, j, k], lut[1, i, j, k], lut[2, i, j, k]))


# Vis
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def vis_3dlut_ploty(lut):
    dim = lut.shape[1]

    idt = identity3d_tensor(dim)

    x, y, z = np.arange(0,dim), np.arange(0,dim), np.arange(0,dim)
    xx, yy, zz = np.meshgrid(x, y, z)
    coords = np.array((xx.ravel(), yy.ravel(), zz.ravel())).T

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]])

    fig.add_trace(
    go.Scatter3d(x=coords[:, 2],
                 y=coords[:, 0],
                 z=coords[:, 1],
                 mode='markers',
                 marker=dict(
                    color=idt.reshape(3,-1).transpose(0,1),
                    opacity=0.5,
                 )),
    row=1, col=1
    )   

    lut = lut.reshape(3,-1).transpose(0,1)

    fig.add_trace(
        go.Scatter3d(x=lut[:, 0],
                    y=lut[:, 1],
                    z=lut[:, 2],
                    mode='markers',
                    marker=dict(
                        color=idt.reshape(3,-1).transpose(0,1),
                        opacity=0.5,
                    )),
        row=1, col=2
    )

def draw_3d(lut, title=None, point_size=30,ax=None):
    if len(lut.shape) == 5:
        lut = lut.squeeze()
    if isinstance(lut, torch.Tensor):
        lut = lut.cpu().numpy()
    
    assert len(lut.shape)==4
    assert lut.shape[0]==3

    dim = lut.shape[1]

    lut = lut.clip(0,1) #[c,b,g,r] [3,dim,dim,dim]

    # lut = lut.reshape(3,-1).transpose(1,0)
    lut_ = lut.reshape(3,-1).T # [N,3]

    if ax is None:
        ax = plt.subplot(111, projection='3d')
    
    ax.set_title(title)

    # ax.set_xlabel('R')
    # ax.set_ylabel('G')
    # ax.set_zlabel("B")

    ax.scatter(lut_[:,0], lut_[:,1], lut_[:,2], c=lut_, s=point_size)

# Tri linear
import torch.nn.functional as F

def trilinear(img,LUT,mode='bilinear',padding_mode='border',align_corners=False):
    '''
        Do trilinear interpolation to image

        Args:
            img: tensor in shape [N,H,W,c] (batch h w channel),shoule normalized in [-1,1]
            LUT: tensor in shape [N,3,dim,dim,dim]

        Result:
            result: interpolated image in [N,C,H,W]
    '''

    assert (len(img.shape)==4)
    assert (img.shape[-1]==3) # C=3
    assert (len(LUT.shape)==5)  # [N,3,B_dim,G_dim,R_dim] [N,C,D,H,W]
    
    # grid_sample expects NxDxHxWx3 (1x1xHxWx3)
    grid = img[:,None] # [N,H,W,C] > [N,D,H,W,C] ~ [N,1,H,W,3]

    # grid sample
    result = F.grid_sample(LUT, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners) # output is [N,C,D,H,W] N:batch size C color D=1 H,W image size

    out_image = result.squeeze(dim=2) #[N,C,D,H,W] > [N,C,H,W] N may equal 1
    
    # if len(out_image.shape) ==3:
    #     out_image = out_image.permute(1,2,0)
    # else:
    #     out_image = out_image.permute(0,2,3,1)

    return out_image