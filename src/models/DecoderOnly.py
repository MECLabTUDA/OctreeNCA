import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
class DecoderOnly(nn.Module):
    def __init__(self, batch_size, extra_channels, device=torch.device("cuda:0"), hidden_size=512):
        super(DecoderOnly, self).__init__()

        self.batch_size = batch_size
        self.extra_channels = extra_channels
        self.device = device
        self.hidden_size = hidden_size

        self.drop0 = nn.Dropout(0.25)

        # Decoder
        
        if False:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_size, hidden_size//2, kernel_size=4, stride=2, padding=1),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.Conv2d(hidden_size//2, hidden_size//2, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.Conv2d(hidden_size//2, hidden_size//2, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.ConvTranspose2d(hidden_size//2, hidden_size//4, kernel_size=4, stride=2, padding=1),  # [batch, 64, 16, 16]
                nn.ReLU(),
                nn.Conv2d(hidden_size//4, hidden_size//4, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.Conv2d(hidden_size//4, hidden_size//4, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.ConvTranspose2d(hidden_size//4, hidden_size//8, kernel_size=4, stride=2, padding=1),  # [batch, 32, 32, 32]
                nn.ReLU(),
                nn.Conv2d(hidden_size//8, hidden_size//8, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.Conv2d(hidden_size//8, hidden_size//8, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.ConvTranspose2d(hidden_size//8, 3, kernel_size=4, stride=2, padding=1),  # [batch, 3, 64, 64]
                #nn.Sigmoid()  # Use sigmoid for normalizing outputs between 0 and 1
            )


        # EMBEDDINGS
        self.vec_dec = nn.Linear(extra_channels, hidden_size*4*4)
        self.vec_dec_1 = nn.Linear(extra_channels, extra_channels*8*8)
        self.vec_dec_2 = nn.Linear(extra_channels, extra_channels*16*16)
        self.vec_dec_3 = nn.Linear(extra_channels, extra_channels*32*32)


        rgb = 0

        self.relu =  nn.ReLU()
        self.convt0 = nn.ConvTranspose2d(hidden_size, hidden_size//2, kernel_size=4, stride=2, padding=1)
        self.convt1 = nn.ConvTranspose2d(hidden_size//2, hidden_size//4, kernel_size=4, stride=2, padding=1)
        self.convt2 = nn.ConvTranspose2d(hidden_size//4, hidden_size//8, kernel_size=4, stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d(hidden_size//8, 3, kernel_size=4, stride=2, padding=1)

        self.conv0 = nn.Conv2d(hidden_size+extra_channels + rgb, hidden_size, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(hidden_size//2+extra_channels + rgb, hidden_size//2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(hidden_size//4+extra_channels + rgb, hidden_size//4, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(hidden_size//8+extra_channels + rgb, hidden_size//8, kernel_size=1, stride=1, padding=0)

        self.conv0_2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(hidden_size//2, hidden_size//2, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(hidden_size//4, hidden_size//4, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = nn.Conv2d(hidden_size//8, hidden_size//8, kernel_size=1, stride=1, padding=0)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.bn0 = nn.BatchNorm2d(hidden_size)
        self.bn1 = nn.BatchNorm2d(hidden_size//2)
        self.bn2 = nn.BatchNorm2d(hidden_size//4)
        self.bn3 = nn.BatchNorm2d(hidden_size//8)


        self.list_backpropTrick = []
        for i in range(batch_size):
            self.list_backpropTrick.append(nn.Conv1d(extra_channels, extra_channels, kernel_size=1, stride=1, padding=0, groups=extra_channels))

    def generate_gaussian_noise(self, image_shape, mean=0.0, std=5.0):
        """
        Generate Gaussian noise.
        :param image_shape: The shape of the image to which the noise will be added.
        :param mean: The mean of the Gaussian distribution.
        :param std: The standard deviation of the Gaussian distribution.
        :return: Gaussian noise with the same shape as the input image.
        """
        noise = torch.normal(mean, std, size=image_shape).to(self.device)
        return noise

    def mix_image_with_noise(self, image, noise_level_range=(0.0, 1.0)):
        """
        Mix the image with Gaussian noise.
        :param image: The input image.
        :param noise_level_range: A tuple indicating the range of noise levels.
                                0.0 means no noise, 1.0 means full noise.
        :return: A mixture of the image and the noise.
        """
        # Generate noise
        noise = self.generate_gaussian_noise(image.shape)

        # Randomly choose a noise level
        noise_level = torch.rand(1).item() * (noise_level_range[1] - noise_level_range[0]) + noise_level_range[0]

        # Interpolate between the image and the noise
        noisy_image = (1 - noise_level) * image + noise_level * noise

        return noisy_image

    def forward(self, x, x_vec_in, **kwargs):

        x_vec_in = x_vec_in.to(self.device)[:, :, None]
        #x_vec_in = self.drop0(x_vec_in)

        # ALLOW BACKPROP THROUGH RANDOM VECTOR
        batch_emb = []
        for i in range(x_vec_in.shape[0]):
            batch_emb.append(self.list_backpropTrick[i].to(self.device)(x_vec_in[i:i+1]))
        #emb = torch.stack(batch_emb, dim=0)
        emb = torch.squeeze(torch.stack(batch_emb, dim=0))

        if x_vec_in.shape[0] == 1:
            emb = emb.to(self.device)[None, :]


        # CREATE NOISY IMAGE
        if self.training:
            noisy_image = self.mix_image_with_noise(x).transpose(1,3)
        else:
            print("TESTING")
            noisy_image = self.generate_gaussian_noise(x.shape).transpose(1,3)

        # RUN MODEL

        dx = self.vec_dec(emb)
        dx = dx.view(dx.shape[0], self.hidden_size, 4, 4)




        #x = self.encoder(x)
        #x = self.decoder(x).transpose(1,3)

        dx = self.relu(dx)
        #print(emb.shape, dx.shape, emb[:, :, None, None].repeat(1, 1, 4, 4).shape)
        dx = self.conv0(torch.cat((dx, emb[:, :, None, None].repeat(1, 1, 4, 4)), dim=1)) #, self.maxPool(self.maxPool(self.maxPool(self.maxPool(noisy_image))))
        dx = self.relu(dx)
        dx = self.conv0_2(dx)
        dx = self.bn0(dx)
        
        dx = self.relu(dx)
        dx = self.convt0(dx)
        dx = self.relu(dx)

        emb1 = self.vec_dec_1(emb)
        emb1 = emb1.view(emb.shape[0], emb.shape[1], 8, 8)
        dx = self.conv1(torch.cat((dx, emb1), dim=1)) #, self.maxPool(self.maxPool(self.maxPool(noisy_image)))
        dx = self.relu(dx)
        dx = self.conv1_2(dx)
        dx = self.bn1(dx)

        dx = self.relu(dx)
        dx = self.convt1(dx)
        dx = self.relu(dx)

        emb2 = self.vec_dec_2(emb)
        emb2 = emb2.view(emb.shape[0], emb.shape[1], 16, 16)
        dx = self.conv2(torch.cat((dx, emb2), dim=1))#, self.maxPool(self.maxPool(noisy_image))
        dx = self.relu(dx)
        dx = self.conv2_2(dx)
        dx = self.bn2(dx)

        dx = self.relu(dx)
        dx = self.convt2(dx)
        dx = self.relu(dx)

        emb3 = self.vec_dec_3(emb)
        emb3 = emb3.view(emb.shape[0], emb.shape[1], 32, 32)
        dx = self.conv3(torch.cat((dx, emb3), dim=1)) #, self.maxPool(noisy_image)
        dx = self.relu(dx)
        dx = self.bn3(dx)
        
        dx = self.conv3_2(dx)
        dx = self.relu(dx)
        dx = self.convt3(dx)

        #x = x*2 -1
        return dx.transpose(1,3)