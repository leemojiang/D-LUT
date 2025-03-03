# Runner Class for Exp2.0

# --- built in ---
import os
import logging
import time
# --- 3rd party ---
import argparse
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# my lib
from .LUT import *
from .models.toy_models import ToyMLP,EnergyScore
from .runner import BasicRunner
from .utils import namespace2dict,imgdir_to_tensor

class Exp2Runner():
    '''
        Exp2.0 Runner class
    '''

    def __init__(self, args:argparse.Namespace):
        '''
            args(argparse.Namespace): argParse namespcae
        
        '''

        assert type(args) == argparse.Namespace,"Wrong args type"

        self.args = args
        # runtime settings
        self.config = argparse.Namespace()

        # setup logs
        self.log_setup()
        # setup torch
        self.torch_setup()

        self.model = None
        self.trainned = False

    def log_setup(self):
        #Set output dir
        run_time =  time.strftime('%Y-%b-%d-%H-%M-%S')
        log_dir = os.path.join(self.args.root_dir,run_time)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir,exist_ok=True)

        self.config.log_dir = log_dir
        self.args.log_dir = log_dir
        self.config.run_time = run_time

        # # logging settings

        # 获取根记录器
        # 禁用根记录器
        logging.getLogger().disabled = True

        # 创建 logger 对象
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.handlers=[]

        # 创建控制台 handler 并设置级别为 DEBUG
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        # 创建文件 handler 并设置级别为 DEBUG
        file_handler = logging.FileHandler(log_dir +'/log.log')
        file_handler.setLevel(logging.DEBUG)

        # 创建 formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 将 formatter 添加到 handlers
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 将 handlers 添加到 logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        #tensor board
        self.config.tb_path = log_dir
        writter = SummaryWriter(log_dir)

        self.writter = writter
        self.logger = logger

    def torch_setup(self):
        # Set device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.device = torch.cuda.current_device()
        self.config.device = self.device
        self.args.device = self.device
        self.logger.info("Using device: {}".format(self.device))

        #set random seeds
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

    @staticmethod
    def gen_dataset(style_dir,style_shape,device=None,add_uniform=False,uniform_percentage=0):
        '''
            Generate dataset from given style image path and shape
        
        Return:
            dataset to feed to dataloader.
        
        '''
        print(f"Style image file: {style_dir}")

        style_image = Image.open(style_dir)
        resized_image = style_image.resize(style_shape)

        print(f"Style image with shape {style_image.size} reshaped as {resized_image.size}")

        rgb_array = np.array(resized_image)

        data = rgb_array.reshape(-1,3).astype(float)
        # ! Normalization
        data = data/255.0
        sample_num = data.shape[0]
        
        assert sample_num == style_shape[0]*style_shape[1]

        if add_uniform:
            extra_data=identity3d_tensor(int((sample_num * uniform_percentage) ** (1/3)))
            extra_data = extra_data.numpy().reshape(-1,3)
            extra_data_num = extra_data.shape[0]

            data = np.concatenate([data,extra_data])

            print(f"Add {extra_data_num} uniform data to {sample_num} datas , with percentage {extra_data_num/sample_num }")
        else:
            print(f"No uniform data add, sample num: {sample_num}")

        data_tensor = torch.tensor(data ,dtype = torch.float32,device=device) #(N,3)

        # Create a TensorDataset from the data
        dataset = TensorDataset(data_tensor)

        return dataset

    @staticmethod
    def model_setup(args):
        model=None
        if args.model_type =="Energy":
            model = EnergyScore(ToyMLP(3,3,swish=args.swish))
        elif args.model_type == "MLP":
            model = ToyMLP(3,3,swish=args.swish)
        else:
            raise ValueError(f"Wrong model type: {args.model_type}")
        
        return model

    def train(self):
        ## Generate dataset
        dataset = Exp2Runner.gen_dataset(self.args.style_dir,self.args.style_shape,device =self.device,add_uniform=self.args.add_uniform,uniform_percentage=self.args.uniform_percentage)

        # Create a DataLoader with shuffling
        train_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        model = Exp2Runner.model_setup(self.args) 
        model = model.to(device=self.device)

        self.logger.info(f"Model type {self.args.model_type}, Swish {self.args.swish}")

        # Don't save model in trainning
        if self.args.save_epoch == None:
            self.args.save_epoch = self.args.train_epochs +1

        BasicRunner.train(self.args,model,train_loader,sigma=self.args.sigma,writter=self.writter,logger=self.logger)

        self.model = model
        # save the nodel after trainnings
        self.config.model_save_dir = Exp2Runner.save_style_model(self.args,self.model)

        self.trainned = True

    @staticmethod
    def save_style_model_path_check(args):
        if not args.save_model_flag:
            print("Model for trainning is not saved.")
            return None
        
        style_name = os.path.splitext(os.path.basename(args.style_dir))[0]

        model_dir = os.path.join(args.root_dir,"models",style_name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir,exist_ok=True)

        model_path = os.path.join(model_dir,f"model_{args.train_epochs}_{args.sigma}.pth")

        if(os.path.isfile(model_path)):
            print("Model exists and")
            return model_path
        else:
            return None

    @staticmethod
    def save_style_model(args,model):
        if not args.save_model_flag:
            print("Model for trainning is not saved.")
            return None
        
        style_name = os.path.splitext(os.path.basename(args.style_dir))[0]

        model_dir = os.path.join(args.root_dir,"models",style_name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir,exist_ok=True)

        model_path = os.path.join(model_dir,f"model_{args.train_epochs}_{args.sigma}.pth")
        
        # assert not os.path.isfile(model_path),"Model exists" 
        if(os.path.isfile(model_path)):
            print("Model exists and will override")

        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}.")
        
        #save config at model_dir
        args_dict = namespace2dict(args)

        with open(os.path.join(model_dir, 'run_args.yml'), 'w') as f:
            yaml.dump(args_dict, f, default_flow_style=False)
        
        # save model dir and style file
        style_dic = {"model_dir": model_path,"style_dir":args.style_dir,"style_shape":args.style_shape}
        with open(os.path.join(model_dir, 'style_config.yml'), 'w') as f:
            yaml.dump(style_dic, f, default_flow_style=False)

        return model_path

    @staticmethod
    def load_style_model(args):
        model = Exp2Runner.model_setup(args)
        if args.model_dir  is None:
            print("No model to load!")
            return None
        else:
            
            model_parentdir = os.path.dirname(args.model_dir)
            if(args.style_dir is None):
                print("No style dir, use model config")
                with open(os.path.join(model_parentdir,"style_config.yml"), 'r') as f:
                    config = yaml.safe_load(f)
                    assert os.path.samefile(config["model_dir"] , args.model_dir)
                    args.style_dir = config["style_dir"]
                    # args.style_shape = config["style_shape"]

            style_name = os.path.splitext(os.path.basename(args.style_dir))[0]
            #model parent dir name:   
            model_dir_name = os.path.basename(model_parentdir)
            assert style_name == model_dir_name 

            model.load_state_dict(torch.load(args.model_dir))
        return model

    def save_config(self):
        # save configs
        with open(os.path.join(self.config.log_dir, 'run_args.yml'), 'w') as f:
        # 将namespace对象转换为字典
            args_dict = namespace2dict(self.args) #这里太诡异了 args_dict修改会直接改变args 
            args_dict['device'] = str(args_dict['device'])
            yaml.dump(args_dict, f, default_flow_style=False)

        with open(os.path.join(self.config.log_dir, 'run_config.yml'), 'w') as f:
        # 将namespace对象转换为字典
            args_dict = namespace2dict(self.config)
            # args_dict['device'] = str(args_dict['device'])
            yaml.dump( args_dict, f, default_flow_style=False)

    def eval(self):
        if (self.model is not None ):
            print(f"Use trainned model! model trainned status: {self.trainned}")
        else:
            if self.args.model_dir is not None:
                self.model = Exp2Runner.load_style_model(self.args)
            else:
                print("No model to eval")
                return
        
        self.config.is_eval = True
        self.model.to(self.device)
        self.model.eval()
        print("Model ready for eval!")

    def visualization(self):
        print("Image Langevin start")
        self.image_langevin()
        print("Image Langevin done")
        if self.image_list:        
            x0 = self.image_list[0].to(self.device)
            for i in range(0,len(self.image_list)):
                nx = self.image_list[i].to(self.device)
                # self.logger.info(f"Langevin steps: {i}")
                nx = torch.clamp(nx,0,1) # After +0.5 should be in 0,1
                dx = nx-x0

                display_image = torch.concat([nx,dx],dim=0) #[N+N,H,W,C]
                self.writter.add_images("Direct Apply Langevie to Image",display_image,i,dataformats='NHWC')

        print("Lut Langevin start")
        self.lut_langevie()
        print("Lut langevie done")

        # to tensor
        img0 = self.content_image_tensor #[N,C,H,W] C=3 [0,1]
        # scale im between -1 and 1 since its used as grid input in grid_sample
        img = (img0 - .5) * 2. #[-1,1]
        img = img.permute(0,2,3,1) # img should in [N,H,W,C]

        img = img.to(self.device) # need gpu
        img0 = img0.to(self.device)

        for i in range(len(self.lut_list)):
            # logger.info(f"LUT at step: {i}")
            lut = self.lut_list[i] # lut [B,dim,dim,dim,3]

            mean_lut = lut.mean(dim=0) #[B,dim,dim,dim,3] > [dim,dim,dim,3]
            vis_lut = mean_lut.permute(3,0,1,2) # > [3,dim,dim,dim]
            # vis LUT
            fig = plt.figure()
            ax = plt.subplot(111, projection='3d')
            draw_3d(vis_lut,ax=ax)
            self.writter.add_figure("Apply Langevin to LUT",fig,i)

            #Apply lut to img
            img_batch = img.shape[0] # img [N,H,W,C]
            tri_lut = vis_lut.expand(img_batch,*vis_lut.shape).to(self.device) #[N,3,dim,dim,dim]
            
            out_img = trilinear(img,tri_lut)

            # vis image
            dx = out_img - img0 #[N,C,H,W]

            display_image = torch.concat([out_img,dx],dim=0) #[2N,C,H,W]

            self.writter.add_images("Apply LUT to Image",display_image,i)

        print("done")


    def save_for_eval(self,save_path,img_list,lut_list=None,content_shape=[300,400],sample_at=40):
        """
            Apply Lut to image and out put to different folders for eval.

            out_dir: Output file parent dir.
            img_list: List of images to process could be path.
            lut_list: Lut list to from langevin steps.
            sample_at: Which steps to use
            content_shape: Content shape in [H,W] [-1,-1] means no reshape

        """
        if lut_list is None:
            lut_list = self.lut_list
            print("Use self lut list")

            if self.lut_list is None:
                raise TypeError("No lut list!!")

        content_list = []
        for img_dir in img_list:
            if os.path.isfile(img_dir):
                content_list.append(img_dir)
            elif os.path.isdir(img_dir):
                for file_name in os.listdir(img_dir):
                    if any(file_name.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']):
                        content_list.append(os.path.join(img_dir,file_name))

        if not os.path.exists(save_path):
            os.makedirs(save_path,exist_ok=True)

        input_path = os.path.join(save_path,"input")
        style_path = os.path.join(save_path,"style")
        result_path = os.path.join(save_path,"result")

        os.makedirs(input_path,exist_ok=True)
        os.makedirs(style_path,exist_ok=True)
        os.makedirs(result_path,exist_ok=True)

        #style img
        style_image= imgdir_to_tensor(self.args.style_dir,self.args.style_shape[::-1]) # style shape is in [W,H]
        style_name = os.path.basename(self.args.style_dir)

        for img_dir in content_list:
            print(f"Process {img_dir}")
            content_name = os.path.basename(img_dir)

            #input img
            img0 = imgdir_to_tensor(img_dir,content_shape) #[N,C,H,W] C=3 [0,1]

            # scale im between -1 and 1 since its used as grid input in grid_sample
            img = (img0 - .5) * 2. #[-1,1]
            img = img.permute(0,2,3,1) # img should in [N,H,W,C]
            img = img.to(self.device)

            #Apply lut #result img
            out_img = Exp2Runner.apply_lut_to_image(img,lut_list,sample_at)

            ## save result
           
            name = f"s_{style_name}-c_{content_name}.png"

            vutils.save_image(img0, os.path.join(input_path,name)) #[N,C,H,W]  # content/input
            vutils.save_image(style_image, os.path.join(style_path,name)) # style
            vutils.save_image(out_img, os.path.join(result_path,name)) # result

        #save config
        dic = {"model_dir":self.args.model_dir, "style_dir":self.args.style_dir, "content_list": content_list}

        with open(os.path.join(save_path, 'save_config.yml'), 'w') as f:
            yaml.dump(dic, f, default_flow_style=False)
        print("done")

        return [input_path,style_path,result_path]

    def vis_and_save(self,save_path_,img_list,lut_list,content_shape=[300,400],sample_step=5,keep_single=True):
        """
            Input: 
                img_list: a list contaning content images file path
                
                content_shape: [H,W]
                
                Keep_single(bool): each saved output only contain single image.

        
        """
        content_list = []

        if keep_single:
            for img_dir in img_list:
                if os.path.isfile(img_dir):
                    content_list.append(img_dir)
                elif os.path.isdir(img_dir):
                    for file_name in os.listdir(img_dir):
                        if any(file_name.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']):
                            content_list.append(os.path.join(img_dir,file_name))
        else:
            content_list=img_list
        
        try:
            self.config.img_list += content_list 
        except:
            self.config.img_list = content_list

        for img_dir in content_list:
            print(f"Process {img_dir}")

            img0 = imgdir_to_tensor(img_dir,content_shape) #[N,C,H,W] C=3 [0,1]

            # scale im between -1 and 1 since its used as grid input in grid_sample
            img = (img0 - .5) * 2. #[-1,1]
            img = img.permute(0,2,3,1) # img should in [N,H,W,C]
            img = img.to(self.device)

            name = os.path.basename(img_dir)
            if img_dir[-1]=="/":
                name = os.path.basename(img_dir[:-1])
            save_path = os.path.join(save_path_,name)

            if not os.path.exists(save_path):
                os.makedirs(save_path,exist_ok=True)
            print(f"File save at {save_path}")
            vutils.save_image(img0, os.path.join(save_path,"origin.png")) #[N,C,H,W]

            for i in range(0,len(lut_list),sample_step):
                out_img = Exp2Runner.apply_lut_to_image(img,lut_list,i)

                vutils.save_image(out_img, os.path.join(save_path,f"{i}.png")) #[N,C,H,W]
                # print(f"Process step {i}")
            print("done")
    
    def save_styleimg(self):
        style_image = Image.open(self.args.style_dir)
        style_name = os.path.basename(self.args.style_dir)
        style_image.save(os.path.join(self.args.log_dir,style_name))

    @staticmethod
    def apply_lut_to_image(img,lut_list,i):
        '''
            Apply langevin Lut list to img.

            Args:
            img0: image tensor [N,H,W,C] in [-1,1]

            lut_list: tensor list. Item/Lut shape [B,dim,dim,dim,3]

            i: step index

            Return:
            Processed single img in [N,C,H,W]
        '''
        lut = lut_list[i] # lut [B,dim,dim,dim,3]

        mean_lut = lut.mean(dim=0) #[B,dim,dim,dim,3] > [dim,dim,dim,3]
        vis_lut = mean_lut.permute(3,0,1,2) # > [3,dim,dim,dim]
        # vis LUT
        
        #Apply lut to img
        img_batch = img.shape[0] # img [N,H,W,C]
        tri_lut = vis_lut.expand(img_batch,*vis_lut.shape) #[N,3,dim,dim,dim]
        out_img = trilinear(img,tri_lut)

        return out_img


    def image_langevin(self):
        self.content_image_tensor = imgdir_to_tensor(self.args.content_dir,self.args.content_shape,max_num=4) # reshape and resized [0,1] [N,C,W,H]

        x0 = self.content_image_tensor.permute(0,2,3,1).to(self.device) # [N,C,H,W] ->> [N,H,W,C]

        image_list = BasicRunner.langevin_dynamics(self.model,x0,L_steps=self.args.L_steps,eps=self.args.eps,keep_path=True)

        self.image_list = image_list
        return image_list
    
    def lut_langevie(self):
        lut = identity3d_tensor(self.args.lut_dim) #[3,dim,dim,dim]
        lut0 = lut.unsqueeze(0).expand(self.args.lut_sample ,*lut.shape) #[B,3,dim,dim,dim]
        lut0= lut0.permute(0,2,3,4,1) #[B,dim,dim,dim,3]
        lut0 = lut0.to(self.device)

        lut_list = BasicRunner.langevin_dynamics(self.model,lut0,L_steps=self.args.L_steps,eps=self.args.eps,keep_path=True) # list of tensor [50,dim,dim,dim,3]

        self.lut_list = lut_list
        return lut_list

    def show_style(self): 
        fig = plt.figure()
        style_image = Image.open(self.args.style_dir)
        resized_image = style_image.resize(self.args.style_shape)
        plt.imshow(resized_image)
        plt.title("Style Image")


        fig2 = plt.figure()
        rgb_array = np.array(resized_image)
        points = rgb_array.reshape(-1,3)/255.0 #[H*W,3] r,g,b order [0,1]  
        ax = plt.subplot(111, projection='3d')
        ax.set_title("Style Color Scatter")
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel("B")
        # ax.scatter(lut_[:,0], lut_[:,1], lut_[:,2], c=lut_, s=point_size)
        ax.scatter(points[:,0], points[:,1], points[:,2], c=points, s=5)
        # return fig
    def show_style_scatter(self,style_shape=[200,150]):
            style_image = Image.open(self.args.style_dir)
            resized_image = style_image.resize(style_shape)
            rgb_array = np.array(resized_image)
            points = rgb_array.reshape(-1,3)/255.0 #[H*W,3] r,g,b order [0,1]

            lut = identity3d_tensor(15)
            l_ponits = lut.reshape(3,-1).T #[3,dim,dim,dim] > [3,dim^3] > [dim^3,3]

            fig = go.Figure(data=[go.Scatter3d(x=l_ponits[:,0], y=l_ponits[:,1], z=l_ponits[:,2],
                                            mode='markers',marker=dict(color=l_ponits,opacity=0.05))])


            fig.add_trace(go.Scatter3d(
                x=points[:,0],
                y=points[:,1],
                z=points[:,2],
                mode='markers',
                marker=dict(
                    size=12,
                    color=points,                # set color to an array/list of desired values
                    # colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )
            ))


            # 更新轴标签
            fig.update_layout(scene=dict(
                xaxis_title='R',
                yaxis_title='G',
                zaxis_title='B'
            ))

            fig.show()

    def show_hist(self):
        score = self.model
        score.to(self.device)

        grids = identity3d_tensor(self.args.vis_dim) # [3,dim,dim,dim]
        grids= grids.permute(1,2,3,0) #[dim,dim,dim,3]
        grids = grids.to(self.device) 

        grad = score(grids) #[dim,dim,dim,3]
        norm = torch.norm(grad,dim=-1).detach().cpu().numpy()

        # numpy版本问题报错
        # writter.add_histogram('Score Norm hist',norm.flatten(),0)
        # No loop matching the specified signature and casting was found for ufunc greater
        fig = plt.figure()
        plt.hist(norm.flatten(),bins=30)
        fig.show()
        self.writter.add_figure("hist fig",fig,0)
        fig2 = plt.figure()
        plt.hist(norm.flatten(),bins=30)
        # return fig2

    def show_score(self):
        score = self.model
        score.to(self.device)

        grids = identity3d_tensor(self.args.vis_dim) # [3,dim,dim,dim]
        grids= grids.permute(1,2,3,0) #[dim,dim,dim,3]
        grids = grids.to(self.device) 

        grad = score(grids) #[dim,dim,dim,3]
        np_grad = grad.to('cpu').detach().numpy()
        np_grids = grids.to('cpu').numpy()

        x,y,z = np_grids[...,0],np_grids[...,1],np_grids[...,2]
        u,v,w = np_grad[...,0],np_grad[...,1],np_grad[...,2]

        color = np_grids.reshape(-1,3) #[N,3]

        fig = go.Figure()
        trace1 =go.Cone(x=x.flatten(), y=y.flatten(), z=z.flatten(),
                                    u=u.flatten(), v=v.flatten(), w=w.flatten(),
                                    sizemode="scaled", showscale = True,sizeref=3,colorscale ='aggrnyl',)
        trace2 = go.Scatter3d(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            mode='markers',
            marker=dict(
                size=5,
                color=color,                # set color to an array/list of desired values
                # colorscale='Viridis',   # choose a colorscale
                opacity=0.5
            )
        )

        fig.add_trace(trace2)
        fig.add_trace(trace2)
        fig.add_trace(trace1)

        # 更新轴标签
        fig.update_layout(scene=dict(
            xaxis_title='R',
            yaxis_title='G',
            zaxis_title='B'
        ))

        fig.show()
        # return fig