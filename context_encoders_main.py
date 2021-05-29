import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader

import torchvision.utils
import torchvision.transforms as transforms

import ce.modules as cemodules

import utils.mask
from utils.metering import Timer, RunningAverage

import random
import matplotlib.pyplot as plt
import numpy as np

import os.path
import shutil


class Args:
    def __init__(self):
        self.seed = None

        self.gpu = True

        self.dataset_path = './dataset/Places365_val_large'

        self.batch_size = 64

        self.load_checkpoint = None
        self.save_checkpoint_dir = './context_encoders_checkpoints'
        self.coarse_checkpoint_step = 15

        self.mode = 'train'  # either 'train' or 'test'

        self.train = 'both'
        #self.train = 'discriminator'
        #self.train = 'generator'

        self.nepochs = 100


def main():
    args = Args()

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    assert(args.mode == 'train' or args.mode == 'test')

    if args.train == 'discriminator':
        print('TRAINING ONLY THE DISCRIMINATOR')

    if args.train == 'generator':
        print('TRAINING ONLY THE GENERATOR')

    if os.path.exists(args.save_checkpoint_dir):
        assert os.path.isdir(args.save_checkpoint_dir)
    else:
        os.makedirs(args.save_checkpoint_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')

    print(f'Device is {device}')

    ################################################
    # Load and initialize model, optimizer and state
    ################################################

    encoder_decoder = cemodules.EncoderDecoder()
    discrim = cemodules.Discriminator()
    encoder_decoder.to(device)
    discrim.to(device)

    lr = 0.0002
    beta1 = 0.5

    history = []

    optimizer_ce = optim.Adam(encoder_decoder.parameters(), lr=10*lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discrim.parameters(), lr=lr, betas=(beta1, 0.999))

    current_epoch = 0

    if args.load_checkpoint:
        if os.path.isfile(args.load_checkpoint):
            print(f'Loading checkpoint file {args.load_checkpoint}')
            checkpoint = torch.load(args.load_checkpoint, map_location=device)

            current_epoch = checkpoint['current_epoch']
            history = checkpoint['history']

            encoder_decoder.load_state_dict(checkpoint['encoder_decoder_state_dict'])
            discrim.load_state_dict(checkpoint['discrim_state_dict'])
            optimizer_ce.load_state_dict(checkpoint['optimizer_ce_state_dict'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        else:
            print(f'Checkpoint file {args.load_checkpoint} not found')

    ##############################
    # Build dataset and DataLoader
    ##############################

    image_size = 128

    transform = transforms.Compose([
        transforms.RandomCrop(image_size, pad_if_needed=True, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full_dataset = utils.mask.MaskedImageFolderCenter(args.dataset_path, transform=transform)
    full_size = len(full_dataset)
    train_size = int(0.96 * full_size)
    train_dataset = Subset(full_dataset, range(train_size))
    test_dataset = Subset(full_dataset, range(train_size, full_size))

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print(f'Training data size is {train_size}')
    print(f'Test data size is {full_size - train_size}')

    ####################################
    # Store and visualize some test data
    ####################################

    imglist = []

    test_data_iterator = iter(test_dl)
    encoder_decoder.eval()

    nimages = 16
    n = 0
    while n < nimages:
        masked_img, center = next(test_data_iterator)
        pred = encoder_decoder(masked_img.to(device))
        for i in range(args.batch_size):
            imglist.append(masked_img[i, :, :, :])
            imglist.append(masked_img[i, :, :, :] + F.pad(F.pad(center[i, :, :, :], (-4, -4, -4, -4)),
                                                          (36, 36, 36, 36), mode='constant', value=0.0))
            imglist.append(masked_img[i, :, :, :] + F.pad(F.pad(pred[i, :, :, :].detach().cpu(), (-4, -4, -4, -4)),
                                                          (36, 36, 36, 36), mode='constant', value=0.0))
            #imglist.append((1 - mask) * pred[0, :, :, :].detach().cpu() + mask * img)
            n += 1
            if n >= nimages:
                break

    plt.figure(figsize=(16, 8))
    plt.axis('off')
    plt.title('MaskedImageFolder')

    plt.imshow(np.transpose(torchvision.utils.make_grid(imglist,
                                                        padding=2, normalize=True, range=(-1, 1), nrow=12), (1, 2, 0)))
    plt.show()

    ###################
    # execute the model
    ###################

    if args.mode == 'train':
        for epoch in range(args.nepochs):
            current_epoch += 1
            train(train_dl, test_dl, encoder_decoder, discrim, optimizer_ce, optimizer_d, device,
                  current_epoch, args, history)

    elif args.mode == 'test':
        validate(test_dl, encoder_decoder, discrim, device, args)
    else:
        print('Invalid execution mode')

##########################
# Define the test function
##########################


@torch.no_grad()
def validate(dataloader, encoder_decoder, discrim, device, args):
    encoder_decoder.eval()
    discrim.eval()

    ljoint = cemodules.EncoderDecoderLoss()
    ladv = torch.nn.BCELoss()

    batch_time = RunningAverage('BatchTime', ':6.3f')
    data_time = RunningAverage('DataTime', ':6.3f')
    discrim_loss_run_avg = RunningAverage('LossD', ':6.3f')
    discrim_real_accuracy_run_avg = RunningAverage('AccD_R', ':6.3f')
    discrim_fake_accuracy_run_avg = RunningAverage('AccD_F', ':6.3f')
    encoder_decoder_loss_run_avg = RunningAverage('LossE', ':6.3f')

    timer = Timer(args.gpu)

    print(f'Testing...')

    for i, (masked_img_batch, gt_center_batch) in enumerate(dataloader):
        masked_img_batch = masked_img_batch.to(device)
        gt_center_batch = gt_center_batch.to(device)

        data_time.update(timer.elapsed())

        batch_size = gt_center_batch.size(0)

        ####################
        # Test discriminator
        ####################

        real_label = torch.ones(batch_size, device=device)
        real_prediction = discrim(gt_center_batch).view(-1)

        discrim_real_loss = ladv(real_prediction, real_label)

        fake_batch = encoder_decoder(masked_img_batch)
        fake_label = torch.zeros(batch_size, device=device)
        fake_prediction = discrim(fake_batch).view(-1)  # detach to avoid mixing computation graphs

        discrim_fake_loss = ladv(fake_prediction, fake_label)

        discrim_loss_run_avg.update(discrim_fake_loss.item() + discrim_real_loss.item())
        discrim_fake_accuracy_run_avg.update(1 - fake_prediction.mean().item())
        discrim_real_accuracy_run_avg.update(real_prediction.mean().item())

        ######################
        # Test encoder-decoder
        ######################

        updated_prediction = discrim(fake_batch).view(-1)
        encoder_decoder_loss = ljoint(fake_batch, gt_center_batch, updated_prediction, real_label)
        encoder_decoder_loss_run_avg.update(encoder_decoder_loss.item())

        batch_time.update(timer.elapsed())

        timer.reset()

    print(f'[Validation batches: {len(dataloader)}] {encoder_decoder_loss_run_avg} | ' +
          f'{discrim_loss_run_avg} | {discrim_real_accuracy_run_avg} | {discrim_fake_accuracy_run_avg} | ' +
          f'{data_time} | {batch_time})')

    return (
        encoder_decoder_loss_run_avg,
        discrim_loss_run_avg,
        discrim_real_accuracy_run_avg,
        discrim_fake_accuracy_run_avg
    )


##############################
# Define the training function
##############################


def train(train_dl, test_dl, encoder_decoder, discrim, optimizer_ce, optimizer_d, device,
          current_epoch, args, history):

    #torch.autograd.set_detect_anomaly(True)
    encoder_decoder.train()
    discrim.train()

    ljoint = cemodules.EncoderDecoderLoss()
    ladv = torch.nn.BCELoss()

    batch_time = RunningAverage('BatchTime', ':6.3f')
    data_time = RunningAverage('DataTime', ':6.3f')
    discrim_loss_run_avg = RunningAverage('LossD', ':6.3f')
    discrim_real_accuracy_run_avg = RunningAverage('AccD_R', ':6.3f')
    discrim_fake_accuracy_run_avg = RunningAverage('AccD_F', ':6.3f')
    encoder_decoder_loss_run_avg = RunningAverage('LossE', ':6.3f')

    timer = Timer(args.gpu)

    #imglist = [[], [], [], []]
    #for k in range(4):
    #    imglist[k].append(test_tensors[2][k, :, :, :].detach().cpu())
    #    imglist[k].append(test_tensors[0][k, :, :, :].detach().cpu()
    #                      + 1 - test_tensors[1][k, :, :, :].detach().cpu())

    print(f'Training epoch {current_epoch}')

    for i, (masked_img_batch, gt_center_batch) in enumerate(train_dl):
        masked_img_batch = masked_img_batch.to(device)
        gt_center_batch = gt_center_batch.to(device)

        data_time.update(timer.elapsed())

        batch_size = gt_center_batch.size(0)

        #####################
        # Train discriminator
        #####################

        discrim.zero_grad()

        real_label = torch.ones(batch_size, device=device)
        real_prediction = discrim(gt_center_batch).view(-1)

        discrim_real_loss = ladv(real_prediction, real_label)

        if args.train == 'discriminator' or args.train == 'both':
            discrim_real_loss.backward()

        fake_batch = encoder_decoder(masked_img_batch)
        fake_label = torch.zeros(batch_size, device=device)
        fake_prediction = discrim(fake_batch.detach()).view(-1)  # detach to avoid mixing computation graphs

        discrim_fake_loss = ladv(fake_prediction, fake_label)

        if args.train == 'discriminator' or args.train == 'both':
            discrim_fake_loss.backward()
            optimizer_d.step()

        with torch.no_grad():
            discrim_loss_run_avg.update(discrim_fake_loss.item() + discrim_real_loss.item())
            discrim_fake_accuracy_run_avg.update(1 - fake_prediction.mean().item())
            discrim_real_accuracy_run_avg.update(real_prediction.mean().item())

        #######################
        # Train encoder-decoder
        #######################

        encoder_decoder.zero_grad()

        #discrim.freeze_parameters()
        updated_prediction = discrim(fake_batch).view(-1)
        #discrim.unfreeze_parameters()

        encoder_decoder_loss = ljoint(fake_batch, gt_center_batch, updated_prediction, real_label)

        if args.train == 'generator' or args.train == 'both':
            encoder_decoder_loss.backward()
            optimizer_ce.step()

        with torch.no_grad():
            encoder_decoder_loss_run_avg.update(encoder_decoder_loss.item())

        optimizer_ce.zero_grad(set_to_none=True)

        batch_time.update(timer.elapsed())

        if i % 5 == 0:
            print(f'[{i}/{len(train_dl)}] {encoder_decoder_loss_run_avg} | ' +
                  f'{discrim_loss_run_avg} | {discrim_real_accuracy_run_avg} | {discrim_fake_accuracy_run_avg} | ' +
                  f'{data_time} | {batch_time})')

        timer.reset()

    #############################
    # Save checkpont at epoch end
    #############################

    with torch.no_grad():
        print('Recording history...')
        val_run_stats = validate(test_dl, encoder_decoder, discrim, device, args)
        snapshot = {
            'train': {
                'encoder_decoder_loss': encoder_decoder_loss_run_avg.avg,
                'discrim_loss': discrim_loss_run_avg.avg,
                'discrim_real_accuracy': discrim_real_accuracy_run_avg.avg,
                'discrim_fake_accuracy': discrim_fake_accuracy_run_avg.avg
            },
            'test': {
                'encoder_decoder_loss': val_run_stats[0],
                'discrim_loss': val_run_stats[1],
                'discrim_real_accuracy': val_run_stats[2],
                'discrim_fake_accuracy': val_run_stats[3]
            }
        }
        history.append(snapshot)

        print('Saving checkpoint...')
        checkpoint_name = f'checkpoint.pth.tar'
        checkpoint_path = os.path.join(args.save_checkpoint_dir, checkpoint_name)
        torch.save({
            'current_epoch': current_epoch,
            'history': history,
            'encoder_decoder_state_dict': encoder_decoder.state_dict(),
            'discrim_state_dict': discrim.state_dict(),
            'optimizer_ce_state_dict': optimizer_ce.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
        }, checkpoint_path)

        if current_epoch % args.coarse_checkpoint_step == 0:
            coarse_checkpoint_path = os.path.join(args.save_checkpoint_dir, f'checkpoint_epoch_{current_epoch}.pth.tar')
            shutil.copy(checkpoint_path, coarse_checkpoint_path)


if __name__ == '__main__':
    main()
