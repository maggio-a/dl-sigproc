import torch
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader

import torchvision.utils
import torchvision.transforms as transforms

import pconv.modules as pcmodules
import pconv.layers as pclayers

import utils.mask
from utils.metering import Timer, RunningAverage

import random
#import matplotlib.pyplot as plt
import numpy as np

import os
import os.path
import shutil


class Args:
    def __init__(self):
        self.seed = None

        self.gpu = True

        self.dataset_path = './dataset/Places365_val_large'

        self.batch_size = 6

        self.load_checkpoint = None
        self.save_checkpoint_dir = './pconv_checkpoints'
        self.coarse_checkpoint_step = 5

        self.mode = 'train'  # either 'train' or 'test'
        self.tuning_phase = False

        self.useAMP = False

        self.nepochs = 100


def main():
    args = Args()

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    assert(args.mode == 'train' or args.mode == 'test')

    if os.path.exists(args.save_checkpoint_dir):
        assert os.path.isdir(args.save_checkpoint_dir)
    else:
        os.makedirs(args.save_checkpoint_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')

    print(f'Device is {device}')
    if torch.cuda.is_available():
        print(f'Running cuda on GPU: {torch.cuda.get_device_name(0)}')

    ################################################
    # Load and initialize model, optimizer and state
    ################################################

    model = pcmodules.UNet()
    model.to(device)

    loss_function = pcmodules.IrregularHolesLoss()
    loss_function.to(device)

    lr = 0.0002

    history = []

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.useAMP)

    current_epoch = 0

    if args.load_checkpoint:
        if os.path.isfile(args.load_checkpoint):
            print(f'Loading checkpoint file {args.load_checkpoint}')
            checkpoint = torch.load(args.load_checkpoint, map_location=device)

            current_epoch = checkpoint['current_epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            history = checkpoint['history']
        else:
            print(f'Checkpoint file {args.load_checkpoint} not found')

    if args.tuning_phase:
        # adjust learning rate
        lr = 0.00005

        # freeze encoding batch normalization
        for block in model.modules():
            if type(block) == pcmodules.UNet.EncoderBlock:
                for layer in block.children():
                    if type(layer) == torch.nn.BatchNorm2d:
                        for parameter in layer.parameters():
                            parameter.requires_grad = False

        # when finetuning clear the optimization state
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=args.useAMP)

    ##############################
    # Build dataset and DataLoader
    ##############################

    image_size = 512
    mask_items = 15

    transform = transforms.Compose([
        transforms.RandomCrop(image_size, pad_if_needed=True, padding_mode='reflect'),
        transforms.ToTensor()
    ])

    full_dataset = utils.mask.MaskedImageFolder(args.dataset_path, mask_generator_items=mask_items, transform=transform)
    full_size = len(full_dataset)
    train_size = int(0.96 * full_size)
    train_dataset = Subset(full_dataset, range(train_size))
    test_dataset = Subset(full_dataset, range(train_size, full_size))

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print(f'Training data size is {train_size}')
    print(f'Test data size is {full_size - train_size}')

    ###################
    # execute the model
    ###################

    if args.mode == 'train':
        for epoch in range(args.nepochs):
            current_epoch += 1
            train(train_dl, test_dl, model, loss_function, optimizer, scaler, device,
                  current_epoch, args, history)
    elif args.mode == 'test':
        validate(test_dl, model, loss_function, device, args)
    else:
        print('Invalid execution mode')


##########################
# Define the test function
##########################


@torch.no_grad()
def validate(dataloader, model, loss_function, device, args) -> RunningAverage:
    model.eval()

    batch_time = RunningAverage('BatchTime', ':6.3f')
    data_time = RunningAverage('DataTime', ':6.3f')
    loss_run_avg = RunningAverage('Loss', ':6.3f')

    timer = Timer(args.gpu)

    print('Testing...')

    for i, (masked_img_batch, mask_batch, gt_batch) in enumerate(dataloader):
        masked_img_batch = masked_img_batch.to(device)
        mask_batch = mask_batch.to(device)
        gt_batch = gt_batch.to(device)

        data_time.update(timer.elapsed())

        prediction, _ = model(masked_img_batch, mask_batch)
        loss_val = loss_function(gt_batch, mask_batch, prediction)

        loss_run_avg.update(loss_val.item(), masked_img_batch.size(0))

        batch_time.update(timer.elapsed())

        timer.reset()

    print(f'[Validation batches: {len(dataloader)}] {loss_run_avg} | {data_time} | {batch_time})')

    return loss_run_avg


##############################
# Define the training function
##############################


def train(train_dl, test_dl, model, loss_function, optimizer, scaler, device, current_epoch, args, history):
    #torch.autograd.set_detect_anomaly(True)

    model.train()

    batch_time = RunningAverage('BatchTime', ':6.3f')
    data_time = RunningAverage('DataTime', ':6.3f')
    loss_run_avg = RunningAverage('Loss', ':6.3f')

    timer = Timer(args.gpu)

    print(f'Training epoch {current_epoch}')

    for i, (masked_img_batch, mask_batch, gt_batch) in enumerate(train_dl):
        masked_img_batch = masked_img_batch.to(device)
        mask_batch = mask_batch.to(device)
        gt_batch = gt_batch.to(device)

        data_time.update(timer.elapsed())

        with torch.cuda.amp.autocast(enabled=args.useAMP):
            prediction, _ = model(masked_img_batch, mask_batch)
            loss_val = loss_function(gt_batch, mask_batch, prediction)

        loss_run_avg.update(loss_val.item(), masked_img_batch.size(0))

        scaler.scale(loss_val).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        batch_time.update(timer.elapsed())

        if i % 10 == 0:
            mem = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) // 1024**2
            print(f'[{i}/{len(train_dl)}] {loss_run_avg} | {data_time} | {batch_time} | Allocated memory: {mem} MBs')

        timer.reset()

    ##############################
    # Save checkpoint at epoch end
    ##############################

    with torch.no_grad():
        print('Recording loss history')
        val_loss_run_avg = validate(test_dl, model, loss_function, device, args)

        history.append({
            'train_loss': loss_run_avg.avg,
            'val_loss': val_loss_run_avg.avg,
            'finetuning': args.tuning_phase
        })
        loss_run_avg.reset()

        print('Saving checkpoint...')
        checkpoint_name = f'checkpoint.pth.tar'
        checkpoint_path = os.path.join(args.save_checkpoint_dir, checkpoint_name)
        torch.save({
            'current_epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'history': history
        }, checkpoint_path)

        if current_epoch % args.coarse_checkpoint_step == 0:
            coarse_checkpoint_path = os.path.join(args.save_checkpoint_dir, f'checkpoint_epoch_{current_epoch}.pth.tar')
            shutil.copy(checkpoint_path, coarse_checkpoint_path)


if __name__ == '__main__':
    main()


