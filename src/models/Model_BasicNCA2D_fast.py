import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import nca_cuda
except:
    pass
 
class BasicNCA2DFast(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False,
                 inplace_relu=False, normalization="batch"):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
                init_method: Weight initialisation function
                kernel_size: defines kernel input size
                groups: if channels in input should be interconnected
        """
        super().__init__()
        self.use_forward_cuda = False

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.inplace_relu = inplace_relu

        # One Input
        self.fc0 = nn.Conv2d(channel_n*2, hidden_size, kernel_size=1)
        self.fc1 = nn.Conv2d(hidden_size, channel_n - input_channels, kernel_size=1, bias=False)
        padding = int((kernel_size-1) / 2)

        self.conv = nn.Conv2d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect", groups=channel_n)
        
        if normalization == "batch":
            self.bn = torch.nn.BatchNorm2d(hidden_size, track_running_stats=False)
        elif normalization == "layer":
            self.bn = torch.nn.LayerNorm(hidden_size)
        elif normalization == "group":
            self.bn = torch.nn.GroupNorm(1, hidden_size)
        elif normalization == "instance":
            self.bn = torch.nn.InstanceNorm2d(hidden_size)
        elif normalization == "none":
            self.bn = torch.nn.Identity()
        else:
            raise ValueError(f"Unknown normalization type {normalization}")
        
        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def update(self, state, fire_rate):
        # state.shape: BCHW
        delta_state = self.conv(state)
        delta_state = torch.cat([state, delta_state], dim=1)
        delta_state = self.fc0(delta_state)
        delta_state = self.bn(delta_state)
        delta_state = F.relu(delta_state)
        delta_state = self.fc1(delta_state)

        if fire_rate is None:
            fire_rate=self.fire_rate

        with torch.no_grad():
            stochastic = torch.zeros([delta_state.size(0),1,delta_state.size(2),
                                    delta_state.size(3)], device=delta_state.device)
            #stochastic = torch.zeros([delta_state.size(0),delta_state.size(1),delta_state.size(2),
            #                        1], device=delta_state.device)
            stochastic.bernoulli_(p=self.fire_rate).float()
        delta_state = delta_state * stochastic

        return state[:, self.input_channels:] + delta_state

    def forward_cuda(self, state: torch.Tensor, steps=10, fire_rate=0.5):
        print("CUDA quick!")
        assert fire_rate == 0.5, "fire_rate must be 0.5 for CUDA implementation"
        assert isinstance(self.bn, torch.nn.modules.linear.Identity), f"{self.bn} not supported in CUDA implementation"
        
        state = state.contiguous()

        for step in range(steps):
            new_state = torch.zeros(state.size(0), state.size(1), state.size(2), state.size(3), device=state.device)
            new_state[:, 0].bernoulli_(0.5)
            nca_cuda.nca2d_cuda(new_state, state, self.conv.weight, self.conv.bias, self.fc0.weight, self.fc0.bias, self.fc1.weight)
            state = new_state
        return state

    def forward(self, x, steps=10, fire_rate=0.5):
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        #x: BHWC
        state = einops.rearrange(x, "b h w c -> b c h w")
        #x: BCHW

        if not self.training and self.use_forward_cuda:
            state = self.forward_cuda(state, steps, fire_rate)
            return einops.rearrange(state, "b c h w -> b h w c")
        

        const_inputs = state[:,0:self.input_channels]

        for step in range(steps):
            new_state = self.update(state, fire_rate)
            state = torch.cat([const_inputs, new_state], dim=1)

        return einops.rearrange(state, "b c h w -> b h w c")
    

