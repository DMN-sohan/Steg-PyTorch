import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StegaStampEncoder(nn.Module):
    def __init__(self, height, width):
        super(StegaStampEncoder, self).__init__()
        # Secret dense layer
        self.secret_dense = nn.Linear(100, 7500)  # Assuming secret_size is 100
        self.secret_dense_activation = nn.ReLU()

        # Convolutional layers
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        # Upsampling layers
        self.up6 = nn.Conv2d(256, 128, 2)
        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.up7 = nn.Conv2d(128, 64, 2)
        self.conv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.up8 = nn.Conv2d(64, 32, 2)
        self.conv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.up9 = nn.Conv2d(32, 32, 2)
        self.conv9 = nn.Conv2d(67, 32, 3, padding=1)
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.residual = nn.Conv2d(32, 3, 1)

    def forward(self, secret, image):
        # Process secret
        secret = secret - 0.5
        secret = self.secret_dense(secret)
        secret = self.secret_dense_activation(secret)
        secret = secret.view(-1, 3, 50, 50)
        secret_enlarged = F.interpolate(secret, scale_factor=8, mode='nearest')

        # Process image
        image = image - 0.5

        # Concatenate secret and image
        x = torch.cat([secret_enlarged, image], dim=1)

        # Encoder
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        conv5 = F.relu(self.conv5(conv4))

        # Decoder
        up6 = F.relu(self.up6(F.interpolate(conv5, scale_factor=2, mode='nearest')))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = F.relu(self.conv6(merge6))
        up7 = F.relu(self.up7(F.interpolate(conv6, scale_factor=2, mode='nearest')))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = F.relu(self.conv7(merge7))
        up8 = F.relu(self.up8(F.interpolate(conv7, scale_factor=2, mode='nearest')))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = F.relu(self.conv8(merge8))
        up9 = F.relu(self.up9(F.interpolate(conv8, scale_factor=2, mode='nearest')))
        merge9 = torch.cat([conv1, up9, x], dim=1)
        conv9 = F.relu(self.conv9(merge9))
        conv10 = F.relu(self.conv10(conv9))
        residual = self.residual(conv9)

        return residual

class StegaStampDecoder(nn.Module):
    def __init__(self, secret_size, height, width):
        super(StegaStampDecoder, self).__init__()
        self.height = height
        self.width = width

        # STN parameters
        self.stn_params = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (height // 8) * (width // 8), 128),
            nn.ReLU()
        )
        self.stn_fc = nn.Linear(128, 6)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (height // 32) * (width // 32), 512),
            nn.ReLU(),
            nn.Linear(512, secret_size)
        )

    def stn(self, x, theta):
        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(x, grid)

    def forward(self, image):
        image = image - 0.5
        stn_params = self.stn_params(image)
        theta = self.stn_fc(stn_params)
        theta = theta.view(-1, 2, 3)
        transformed_image = self.stn(image, theta)
        return self.decoder(transformed_image)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, image):
        x = image - 0.5
        x = self.model(x)
        output = torch.mean(x)
        return output, x

def get_secret_acc(secret_true, secret_pred):
    secret_pred = torch.round(torch.sigmoid(secret_pred))
    correct_pred = secret_pred.size(1) - torch.count_nonzero(secret_pred - secret_true, dim=1)
    
    str_acc = 1.0 - torch.count_nonzero(correct_pred - secret_pred.size(1)).float() / correct_pred.size(0)
    bit_acc = torch.sum(correct_pred).float() / secret_pred.numel()
    
    return bit_acc, str_acc

# The following functions need to be implemented in utils.py:
# - transform_net
# - jpeg_compress_decompress
# These functions require more complex transformations and will be included in the utils.py file.

def build_model(encoder, decoder, discriminator, secret_input, image_input, l2_edge_gain, borders,
                secret_size, M, loss_scales, yuv_scales, args, global_step):
    # This function needs to be implemented in the training script (train.py)
    # as it involves the training process and loss calculations.
    pass

def prepare_deployment_hiding_graph(encoder, secret_input, image_input):
    residual = encoder(secret_input, image_input)
    encoded_image = residual + image_input
    encoded_image = torch.clamp(encoded_image, 0, 1)
    return encoded_image, residual

def prepare_deployment_reveal_graph(decoder, image_input):
    decoded_secret = decoder(image_input)
    return torch.round(torch.sigmoid(decoded_secret))
