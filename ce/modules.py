import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 4000, 4, 1, 0)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4000, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
        self.reset_weights()

    def reset_weights(self):
        self.apply(init_weights)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.d = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )
        self.reset_weights()

    def reset_weights(self):
        self.apply(init_weights)

    def forward(self, x):
        return self.d(x)

    def freeze_parameters(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze_parameters(self):
        for parameter in self.parameters():
            parameter.requires_grad = True


class EncoderDecoderLoss(torch.nn.Module):

    def __init__(self):
        super(EncoderDecoderLoss, self).__init__()
        self.l2 = torch.nn.MSELoss()
        self.ladv = torch.nn.BCELoss()
        self.lambda_rec = 0.999

    def forward(self, fake_batch, gt_center_batch, adversarial_prediction, adversarial_label):
        device = fake_batch.device
        weight = F.pad(torch.ones((fake_batch.size(0), fake_batch.size(1), 56, 56), device=device),
                       (4, 4, 4, 4), mode='constant', value=10.0)
        return (self.lambda_rec * torch.mul(weight, torch.square(gt_center_batch - fake_batch)).mean() +
                (1 - self.lambda_rec) * self.ladv(adversarial_prediction, adversarial_label))

        # Convolve with a 3x3 1-kernel to dilate the non-hole region and compute weights
        # Pixels near the hole border contribute to the reconstruction loss with larger weight (10x)

        #with torch.no_grad():
        #    shrunk_mask_batch = F.conv2d(mask_batch, torch.ones((3, 3, 3, 3), device=device), stride=1, padding=1)
        #    shrunk_mask_batch = torch.clamp(shrunk_mask_batch, 0, 1)
        #    weight = 10 * (1 - mask_batch) - 9 * (1 - shrunk_mask_batch)

        #return (self.lambda_rec * self.l2(weight * gt_batch, weight * fake_batch) +
        #        self.lambda_adv * self.ladv(adversarial_prediction, adversarial_label))
