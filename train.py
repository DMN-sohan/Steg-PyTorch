import argparse
import glob
import os
from PIL import Image, ImageOps
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import utils
from models import StegaStampEncoder, StegaStampDecoder, Discriminator, get_secret_acc

TRAIN_PATH = './data/mirflickr/images1/images/'
LOGS_PATH = "./logs/"
CHECKPOINTS_PATH = './checkpoints/'
SAVED_MODELS = './saved_models'

if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)

class SteganographyDataset(Dataset):
    def __init__(self, files_list, secret_size, size=(400, 400)):
        self.files_list = files_list
        self.secret_size = secret_size
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        img_cover_path = self.files_list[idx]
        try:
            img_cover = Image.open(img_cover_path).convert("RGB")
            img_cover = self.transform(img_cover)
        except:
            img_cover = torch.zeros((3, self.size[0], self.size[1]))

        secret = torch.bernoulli(torch.ones(self.secret_size) * 0.5)
        return img_cover, secret

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--secret_size', type=int, default=20)
    parser.add_argument('--num_steps', type=int, default=140000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.0001)
    # Add other arguments as needed

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files_list = glob.glob(os.path.join(TRAIN_PATH, "**/*"))
    dataset = SteganographyDataset(files_list, args.secret_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    encoder = StegaStampEncoder(height=400, width=400).to(device)
    decoder = StegaStampDecoder(secret_size=args.secret_size, height=400, width=400).to(device)
    discriminator = Discriminator().to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=.00001)

    criterion = nn.BCEWithLogitsLoss()

    global_step = 0
    while global_step < args.num_steps:
        for images, secrets in dataloader:
            images = images.to(device)
            secrets = secrets.to(device)

            # Train encoder and decoder
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            residual = encoder(secrets, images)
            encoded_images = residual + images
            encoded_images = torch.clamp(encoded_images, 0, 1)

            decoded_secrets = decoder(encoded_images)

            secret_loss = criterion(decoded_secrets, secrets)

            if not args.no_gan:
                D_output_fake, _ = discriminator(encoded_images)
                G_loss = -D_output_fake.mean()
            else:
                G_loss = torch.tensor(0.0).to(device)

            encoder_decoder_loss = secret_loss + G_loss
            encoder_decoder_loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            # Train discriminator
            if not args.no_gan:
                discriminator_optimizer.zero_grad()
                D_output_real, _ = discriminator(images)
                D_output_fake, _ = discriminator(encoded_images.detach())
                D_loss = D_output_fake.mean() - D_output_real.mean()
                D_loss.backward()
                discriminator_optimizer.step()

                # Clip discriminator weights
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            global_step += 1

            if global_step % 100 == 0:
                bit_acc, str_acc = get_secret_acc(secrets, decoded_secrets)
                print(f"Step {global_step}: Secret Loss: {secret_loss.item():.4f}, Bit Accuracy: {bit_acc:.4f}, String Accuracy: {str_acc:.4f}")

            if global_step % 10000 == 0:
                torch.save({
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'global_step': global_step
                }, os.path.join(CHECKPOINTS_PATH, args.exp_name, f"{args.exp_name}.pth"))

            if global_step >= args.num_steps:
                break

    # Save final model
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'discriminator': discriminator.state_dict(),
        'global_step': global_step
    }, os.path.join(SAVED_MODELS, f"{args.exp_name}.pth"))

if __name__ == "__main__":
  main()
