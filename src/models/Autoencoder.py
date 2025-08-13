import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
class Autoencoder(nn.Module):
    def __init__(self, batch_size, extra_channels, device=torch.device("cuda:0"), hidden_size=512):
        super(Autoencoder, self).__init__()

        self.batch_size = batch_size
        self.extra_channels = extra_channels
        self.device = device
        self.list_backpropTrick = []
        for i in range(batch_size):
            self.list_backpropTrick.append(nn.Conv1d(extra_channels, extra_channels, kernel_size=1, stride=1, padding=0, groups=extra_channels).to(self.device))

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [batch, 64, 32, 32]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [batch, 128, 16, 16]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [batch, 256, 8, 8]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 32), # Latent space
            nn.ReLU()
        )



        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [batch, 128, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [batch, 64, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # [batch, 3, 128, 128]
            nn.Sigmoid()
        )


    def forward(self, x, x_vec_in, *args, **kwargs):
        #x_vec_in = x_vec_in.to(self.device)
        #plt.imshow(x[0, :, :, 0:3].real.detach().cpu().numpy())
        #plt.show()

        x = x.transpose(1,3).to(self.device)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.transpose(1,3)
        return x