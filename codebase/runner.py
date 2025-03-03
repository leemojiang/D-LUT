import torch
import torch.optim as optim
import numpy as np
import os

class AnnealRunner():
    @staticmethod
    def anneal_langevin_dynamics(score_fn,x0,epss,L_steps_each,vis_callback=None,vis_steps=10,keep_path=False):
        '''
            Basic langevin sampling
        
        Args:
            score_fn: score model
            x0 (tensor) : start point in shape [...,x] x should equal score model input dim.

            epss: noise level list 
            L_steps_each: steps in each noise level

            vis_callback: function in f(xn,n) n is step size
            vis_steps: call back interval

            keep_path: If true return [x0,x1...xn] in list.
        '''

        x_list=[x0]
        x=x0
        step = 0

        for eps in epss:
            for i in range(L_steps_each):
                step += 1
                x = x + eps/2. * score_fn(x).detach()
                x = x + torch.randn_like(x) * np.sqrt(eps)

                if step % vis_steps ==0  and vis_callback is not None:
                    vis_callback(x,step,eps)

                if keep_path:
                    x_list.append(x)
        
        if keep_path:
            return x_list
        else:
            return x







class BasicRunner():
    '''
        Runner class for EXP1.31
    '''

    def __init__(self, args):
        self.args = args
    
    @staticmethod
    def train(args,model,train_loader,sigma=0.1,writter=None,logger=None):
        '''
            Basic trainning function.

            Args:
                args: using args.device args.train_epochs args.learning_rate
                sigma (float): noise level. 
                
                model: score model. Default optimizer is Adam optimizer.
                train_loader: trainning datasets loader
                
                logger: logging handle
                writter: tensorboard writter handle
        '''

        num_epochs = args.train_epochs
        learning_rate = args.learning_rate
       
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(num_epochs):
            all_losses=[]  # initialize the total loss for the epoch
            for batch,[x] in enumerate(train_loader):
                # Zero the parameter gradients
                # optimizer.zero_grad()

                x = x.requires_grad_().to(device=args.device) #[b,2]
                v = torch.randn_like(x, device=args.device)#[b,2]
                v = v * sigma # That's the bug!!
                # DMS loss
                x_ = x + v
                s = model(x_)#[b,dim]
                loss = torch.norm(s + v/(sigma**2), dim=-1)**2 # [b]
                loss = loss.mean()/2. #[1]

                # Backward pass and optimization
                loss.backward()

                #Clip norm
                # grad = nn.utils.clip_grad_norm_(self.model.parameters(), self.clipnorm)

                optimizer.step()
                optimizer.zero_grad()
                all_losses.append(loss.item()) # to scalar

            if logger:
                average_loss = np.mean(all_losses).astype(np.float32)  # calculate the average loss for the epoch
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}")
            if writter:
                #tb writter
                writter.add_scalar(f'train/loss',loss,epoch)

            # save model
            if (epoch+1) % args.save_epoch == 0:
                model_dir = os.path.join(args.log_dir,"models")
                
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir,exist_ok=True)


                model_path = os.path.join(model_dir,f"model_{epoch}.pth")
                assert not os.path.isfile(model_path),"Model exists" 
                torch.save(model.state_dict(), model_path)
                
                print(f"Trainning done model saved at {epoch}")

                if logger:
                    logger.info(f"Model saved at epoch {epoch+1}.")

    @staticmethod
    def langevin_dynamics(score_fn,x0,L_steps,eps,vis_callback=None,vis_steps=10,keep_path=False):
        '''
            Basic langevin sampling
        
        Args:
            score: score model
            x0 (tensor) : start point in shape [...,x] x should equal score model input dim.

            L_steps : Langevin steps.
            eps: noise level

            vis_callback: function in f(xn,n) n is step size
            vis_steps: call back interval

            keep_path: If true return [x0,x1...xn] in list.
        '''
        x_list=[x0]
        x=x0
        for i in range(L_steps):
            x = x + eps/2. * score_fn(x).detach()
            x = x + torch.randn_like(x) * np.sqrt(eps)

            if i % vis_steps ==0  and vis_callback is not None:
                vis_callback(x,i)

            if keep_path:
                x_list.append(x.to("cpu"))
        
        if keep_path:
            return x_list
        else:
            return x
        
    @staticmethod
    def not_langevin_dynamics(score_fn,x0,L_steps,eps,vis_callback=None,vis_steps=10,keep_path=False):
        '''
            根据3-24会议的建议 尝试一下去掉random的Langevin动力学
        
        Args:
            score: score model
            x0 (tensor) : start point in shape [...,x] x should equal score model input dim.

            L_steps : Langevin steps.
            eps: noise level

            vis_callback: function in f(xn,n) n is step size
            vis_steps: call back interval

            keep_path: If true return [x0,x1...xn] in list.
        '''
        x_list=[x0]
        x=x0
        for i in range(L_steps):
            x = x + eps/2. * score_fn(x).detach()
            # x = x + torch.randn_like(x) * np.sqrt(eps)

            if i % vis_steps ==0  and vis_callback is not None:
                vis_callback(x,i)

            if keep_path:
                x_list.append(x)
        
        if keep_path:
            return x_list
        else:
            return x


class BasicEnergyRunner():
    '''
        Runner class for EXP1.31
    '''

    def __init__(self, args):
        self.args = args
    
    @staticmethod
    def train(args,model,train_loader,sigma=0.1,writter=None,logger=None):
        '''
            Basic trainning function.

            Args:
                args: using args.device args.train_epochs args.learning_rate
                sigma (float): noise level. 
                
                model: score model. Default optimizer is Adam optimizer.
                train_loader: trainning datasets loader
                
                logger: logging handle
                writter: tensorboard writter handle
        '''

        num_epochs = args.train_epochs
        learning_rate = args.learning_rate
       
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(num_epochs):
            all_losses=[]  # initialize the total loss for the epoch
            for batch,[x] in enumerate(train_loader):
                # Zero the parameter gradients
                # optimizer.zero_grad()

                x = x.requires_grad_().to(device=args.device) #[b,2]
                v = torch.randn_like(x, device=args.device)#[b,2]
                v = v * sigma # That's the bug!!
                # DMS loss
                x_ = x + v
                s = model.score(x_)#[b,dim]
                loss = torch.norm(s + v/(sigma**2), dim=-1)**2 # [b]
                loss = loss.mean()/2. #[1]

                # Backward pass and optimization
                loss.backward()

                #Clip norm
                # grad = nn.utils.clip_grad_norm_(self.model.parameters(), self.clipnorm)

                optimizer.step()
                optimizer.zero_grad()
                all_losses.append(loss.item()) # to scalar

            if logger:
                average_loss = np.mean(all_losses).astype(np.float32)  # calculate the average loss for the epoch
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}")
            if writter:
                #tb writter
                writter.add_scalar(f'train/loss',loss,epoch)

            # save model
            if (epoch+1) % args.save_epoch == 0:
                model_dir = os.path.join(args.log_dir,"models")
                
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir,exist_ok=True)


                model_path = os.path.join(model_dir,f"model_{epoch}.pth")
                assert not os.path.isfile(model_path),"Model exists" 
                torch.save(model.state_dict(), model_path)
                
                print(f"Trainning done model saved at {epoch}")

                if logger:
                    logger.info(f"Model saved at epoch {epoch+1}.")

    @staticmethod
    def langevin_dynamics(score_fn,x0,L_steps,eps,vis_callback=None,vis_steps=10,keep_path=False):
        '''
            Basic langevin sampling
        
        Args:
            score: score model
            x0 (tensor) : start point in shape [...,x] x should equal score model input dim.

            L_steps : Langevin steps.
            eps: noise level

            vis_callback: function in f(xn,n) n is step size
            vis_steps: call back interval

            keep_path: If true return [x0,x1...xn] in list.
        '''
        x_list=[x0]
        x=x0
        for i in range(L_steps):
            x = x + eps/2. * score_fn(x).detach()
            x = x + torch.randn_like(x) * np.sqrt(eps)

            if i % vis_steps ==0  and vis_callback is not None:
                vis_callback(x,i)

            if keep_path:
                x_list.append(x)
        
        if keep_path:
            return x_list
        else:
            return x