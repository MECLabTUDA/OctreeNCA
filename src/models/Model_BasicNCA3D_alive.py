import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.Model_BasicNCA3D import BasicNCA3D
 
class BasicNCA3D_alive(BasicNCA3D):

    def alive(self, x):
        #print("SHAPE", F.max_pool3d(x[:, self.input_channels:self.input_channels+1, ...], kernel_size=7, stride=1, padding=3).shape)
        return F.max_pool3d(F.max_pool3d(x[:, self.input_channels:self.input_channels+1, ...], kernel_size=7, stride=1, padding=3), kernel_size=7, stride=1, padding=3) > 0
        #return F.max_pool3d(x[:, self.input_channels:self.input_channels+1, ...], kernel_size=7, stride=1, padding=3) > 0

    def update(self, x_in, fire_rate):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image
                fire_rate: random activation of cells
        """
        x = x_in.transpose(1,4)
        life_mask = self.alive(x)

        dx = self.perceive(x)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])>fire_rate
        #stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),dx.size(4)])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic
        dx = dx.transpose(1,4)

        dx = dx * life_mask

        x = x+dx

        #print(x.shape)
        #post_life_mask = self.alive(x)
        #life_mask = (pre_life_mask & post_life_mask).float()
        #x = x * life_mask

        x = x.transpose(1,4)

        return x