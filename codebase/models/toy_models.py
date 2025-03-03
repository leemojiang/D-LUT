# Ref https://github.com/Ending2015a/toy_gradlogp
import torch
from torch import nn
import os

# --- primitives ---
class Swish(nn.Module):
    def __init__(self, dim=-1):
        """Swish activ bootleg from
        https://github.com/wgrathwohl/LSD/blob/master/networks.py#L299

        Args:
            dim (int, optional): input/output dimension. Defaults to -1.
        """
        super().__init__()
        if dim > 0:
            self.beta = nn.Parameter(torch.ones((dim,)))
        else:
            self.beta = torch.ones((1,))

    def forward(self, x):
        if len(x.size()) == 2:
            return x * torch.sigmoid(self.beta[None, :] * x)
        else:
            return x * torch.sigmoid(self.beta[None, :, None, None] * x)

class ToyMLP(nn.Module):
    def __init__(
        self, 
        input_dim=2,
        output_dim=1,
        units=[300, 300],
        swish=True,
        dropout=False
    ):
        """Toy MLP from
        https://github.com/ermongroup/ncsn/blob/master/runners/toy_runner.py#L198

        Args:
            input_dim (int, optional): input dimensions. Defaults to 2.
            output_dim (int, optional): output dimensions. Defaults to 1.
            units (list, optional): hidden units. Defaults to [300, 300].
            swish (bool, optional): use swish as activation function. Set False to use
                soft plus instead. Defaults to True.
            dropout (bool, optional): use dropout layers. Defaults to False.
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in units:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                Swish(out_dim) if swish else nn.Softplus(),
                nn.Dropout(.5) if dropout else nn.Identity()
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        shape = x.shape
        x= x.reshape(-1,shape[-1])
        return self.net(x).reshape(*shape)
    
# --- energy model ---
class EnergyScore(nn.Module):
    def __init__(self, net):
        """A simple energy model

        Args:
            net (nn.Module): An energy function, the output shape of
                the energy function should be (b, 1). The score is
                computed by grad(-E(x))

            x should be like [...,dim]
            score output is the same as input x 

            b for batchsize 
        """
        super().__init__()
        self.net = net # energy net

    def forward(self, x):
        shape = x.shape
        x= x.reshape(-1,shape[-1]) # 这个解决了require_grad的报错.... 神奇
        return self.score(x).reshape(*shape)
    
    def energy(self,x):
        return self.net(x)

    def score(self, x, sigma=None):
        x = x.requires_grad_()
        logp = -self.net(x).sum()
        return torch.autograd.grad(logp, x, create_graph=True)[0]
        # return torch.autograd.grad(logp, x, create_graph=False)[0] # only need gradient that not work because Need hessian for optim
    

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self