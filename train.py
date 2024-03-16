import dataset.data_loader.VideoFramesDataset
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
import torch
import argparse
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
import torch.nn as nn
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
# from Utils.get_files_from_split import get_files_from_split
import pandas as pd
from dataset.transforms import NormalizeVideo, ToTensorVideo
import numpy as np
from neural_methods.trainer import PhysnetTrainer
import yaml
from config import get_config


def get_files_from_splits(splits_file):
    video_files = []
    with open(splits_file, 'r') as f:
        for line in f:
            parts = line.split(" ")
            if len(parts) == 2:
                video_path, label = parts
                video_files.append(video_path)
    return video_files


def parse_args():
    parser = argparse.ArgumentParser(description="Train and test PhysNet model.")
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the configuration file.')
    args = parser.parse_args()
    return args

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':
    # Define the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Version ", torch.__version__)
    # Print the device being used
    print("Using device:", device)

    model = PhysNet_padding_Encoder_Decoder_MAX(frames=32)

    model = model.to(device)

    pretrained_weights = torch.load("Exp1/SCAMPS_PhysNet_DiffNormalized.pth")
    if pretrained_weights:
        model.load_state_dict(pretrained_weights, strict=False)
        print("Pretrained Loaded")

    for param in model.parameters():
        param.requires_grad = True

    parser = argparse.ArgumentParser()
    #parser = add_args(parser)
    parser.add_argument('--config_file', required=False,
                        default="config.yaml", type=str, help="The name of the model.")
    
    # parser = Dataset.BaseLoader.BaseLoader.add_data_loader_args(parser)

    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')


    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        ToTensorVideo(), # scales pixels from [0, 255] to [0, 1]
    ])


    # train_splits=pd.read_json('Dataset/splits/train.json', dtype=False)

    # train_fake_files,  train_real_files = get_files_from_split(train_splits)

    # valid_splits=pd.read_json('Dataset/splits/val.json', dtype=False)

    # valid_fake_files, valid_real_files = get_files_from_split(valid_splits)


    train_real_files = get_files_from_splits('/vol/research/DeepFakeDet/notebooks/FaceForensics++/original_sequences/youtube/c23/train10.txt')
    train_fake_files = get_files_from_splits('/vol/research/DeepFakeDet/notebooks/FaceForensics++/manipulated_sequences/NeuralTextures/c23/train10.txt')

    valid_real_files = get_files_from_splits('/vol/research/DeepFakeDet/notebooks/FaceForensics++/original_sequences/youtube/c23/val10.txt')
    valid_fake_files = get_files_from_splits('/vol/research/DeepFakeDet/notebooks/FaceForensics++/manipulated_sequences/NeuralTextures/c23/val10.txt')

    train_dataset = dataset.data_loader.VideoFramesDataset.VideoFramesDataset(
        train_real_files,
        train_fake_files,
        frames_per_clip=32,
        transform=transform,
    )

    valid_dataset = dataset.data_loader.VideoFramesDataset.VideoFramesDataset(
        valid_real_files,
        valid_fake_files,
        frames_per_clip=32,
        transform=transform,
    )


    # check the sampling frames
    # TODO: sanity check on the dataset samples
    for i, s in enumerate(train_dataset):
        sample, label, video_idx = s
        print(sample.min(), sample.max(), sample.size())
        print(label.min(), label.max(), label.size())
        if i == 10:
            break

    print("Train dataset length, ", len(train_dataset))
    print("Valid dataset length, ", len(valid_dataset))


    print("Load data..")
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=12, shuffle=False, num_workers=8)

    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)
    # momentum = 0.9
    # weight_decay = 0.00001
    # optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optimiser = SGD(model.parameters(), lr=0.0001, momentum=momentum, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=2, min_lr=0.0000000009, verbose=True)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.0001, epochs=50, steps_per_epoch=len(train_loader))
    print("Finished loading data..")
    twriter = SummaryWriter(log_dir='Exp1/logs/train/test2')
    vwriter = SummaryWriter(log_dir='Exp1/logs/train/test2')


    print("Start training..")
    data_loaders = {"train": train_loader, "valid": valid_loader}
    physnet_trainer = PhysnetTrainer(config, data_loaders)
    physnet_trainer.train(data_loaders)