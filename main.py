""" Codebase refactored from Code Base by https://github.com/ubicomplab/rPPG-Toolbox."""

import argparse
import random
import time
from torchvision import transforms
import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from neural_methods.trainer.PhysnetrppgTrainer import PhysnetrppgTrainer
from neural_methods.trainer.ResNet3DTrainer import ResNet3DTrainer
from neural_methods.trainer.XceptionNetTrainer import XceptionNetTrainer
from neural_methods.trainer.Xception3DTrainer import Xception3DTrainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader
import dataset.data_loader.ImageDataLoader
import dataset.data_loader.VideoFramesDataset
from dataset.transforms import NormalizeVideo, ToTensorVideo, ToTensorVideoNoPermutation
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from neural_methods.trainer.MultiPhysNetModelTrainer import MultiPhysNetModelTrainer

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml", type=str, help="The name of the model.")
    '''Neural Method Sample YAML LIST:
      SCAMPS_SCAMPS_UBFC-rPPG_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_PHYSNET_BASIC.yaml
      SCAMPS_SCAMPS_PURE_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_PURE_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_PURE_PHYSNET_BASIC.yaml
      PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml
      PURE_PURE_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      PURE_PURE_UBFC-rPPG_PHYSNET_BASIC.yaml
      PURE_PURE_MMPD_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_DEEPPHYS_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_PHYSNET_BASIC.yaml
      MMPD_MMPD_UBFC-rPPG_TSCAN_BASIC.yaml
    Unsupervised Method Sample YAML LIST:
      PURE_UNSUPERVISED.yaml
      UBFC-rPPG_UNSUPERVISED.yaml
    '''
    return parser

def get_rPPG(config, data_loader_dict):
    if config.MODEL.NAME == "Physnet":
        model_trainer = PhysnetrppgTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.get_rPPG(data_loader_dict)

def train_and_test(config, data_loader_dict):

    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
        unique_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        writer = SummaryWriter(log_dir=f'Exp1/logs/output/PhysNet/{config.TRAIN.MODEL_FILE_NAME}_{unique_time}')
    elif config.MODEL.NAME == "ResNet3D":
        model_trainer = ResNet3DTrainer(config, data_loader_dict)
        unique_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        writer = SummaryWriter(log_dir=f'Exp1/logs/output/ResNet3D/{config.TRAIN.MODEL_FILE_NAME}_{unique_time}')
    elif config.MODEL.NAME == "Xception":
        model_trainer = XceptionNetTrainer(config, data_loader_dict)
        unique_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        writer = SummaryWriter(log_dir=f'Exp1/logs/output/XceptionNet/{config.TRAIN.MODEL_FILE_NAME}_{unique_time}')
    elif config.MODEL.NAME == "Xception3D":
        model_trainer = Xception3DTrainer(config, data_loader_dict)
        unique_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        writer = SummaryWriter(log_dir=f'Exp1/logs/output/XceptionNet3D/{config.TRAIN.MODEL_FILE_NAME}_{unique_time}')
    elif config.MODEL.NAME == "MultiPhysNetModel":
        model_trainer = MultiPhysNetModelTrainer(config, data_loader_dict)
        unique_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        writer = SummaryWriter(log_dir=f'Exp1/logs/output/MultiPhysNetModelTrainer/{config.TRAIN.MODEL_FILE_NAME}_{unique_time}')    
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.train(data_loader_dict, writer)
    # test on same DS
    # model_trainer.test(data_loader_dict, twriter, vwriter)
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict):
    """Tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "ResNet3D":
        model_trainer = trainer.ResNet3DTrainer.ResNet3DTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Xception":
        model_trainer = XceptionNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Xception3D":
        model_trainer = trainer.Xception3DTrainer.Xception3DTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "MultiPhysNetModel":
        model_trainer = MultiPhysNetModelTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')

    model_trainer.test(data_loader_dict)


def get_files_from_splits(splits_file):
    video_files = []
    with open(splits_file, 'r') as f:
        for line in f:
            parts = line.split(" ")
            if len(parts) == 2:
                video_path, label = parts
                video_files.append(video_path)
    return video_files

if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')
    data_loader_dict = dict() # dictionary of data loaders 
    
    #transforms
    if config.MODEL.NAME == "Physnet" or config.MODEL.NAME == "MultiPhysNetModel":
        print("Entered for transforming..")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            ToTensorVideo(), # scales pixels from [0, 255] to [0, 1]
        ])
    elif config.MODEL.NAME == "ResNet3D":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            ToTensorVideo(), # scales pixels from [0, 255] to [0, 1]
        ])
    elif config.MODEL.NAME == "Xception":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            ToTensorVideo(), # scales pixels from [0, 255] to [0, 1]
        ])
    elif config.MODEL.NAME == "Xception3D":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            ToTensorVideo(), # scales pixels from [0, 255] to [0, 1]
        ]) 
    elif config.MODEL.NAME == "Physnetrppg":
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            ToTensorVideo(),
        ]) 

    if (config.MODEL.NAME == "MultiPhysNetModel"):
            # file contains 32 frames skipping 2: rPPG training labels - real
            rPPG_train_path='rppgpredictions_16frames_train_final.csv'
            rPPG_valid_path='rppgpredictions_16frames_valid_final.csv'
            rPPG_test_path='rppgpredictions_16frames_test_final.csv'
    else:
            rPPG_train_path=None
            rPPG_valid_path=None
            rPPG_test_path=None

    if config.TOOLBOX_MODE == "train_and_test":
        print("Entered train and test")

        # get 'real' video frame filepaths
        train_real_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/original_sequences/youtube/c23/train.txt')
        valid_real_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/original_sequences/youtube/c23/val.txt')

        train_dataset_names = config.TRAIN.DATA.DATASET.replace(" ", "").split(",")
        valid_dataset_names = config.VALID.DATA.DATASET.replace(" ", "").split(",")

        train_fake_dataset_file_paths = {}
        valid_fake_dataset_file_paths = {}

        # get 'fake' video frame filepaths. put multiple datasets in a dictionary 
        for train_dataset in train_dataset_names:
            train_fake_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/manipulated_sequences/{train_dataset}/c23/train.txt')
            train_fake_dataset_file_paths[train_dataset] = train_fake_files
        
        for valid_dataset in valid_dataset_names:
            valid_fake_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/manipulated_sequences/{valid_dataset}/c23/val.txt')
            valid_fake_dataset_file_paths[valid_dataset] = valid_fake_files
            
        # Create and initialize the train dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset paths
        if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):
            print("--- Loading training data ---")
            if config.MODEL.NAME == "Xception":
                train_loader = dataset.data_loader.ImageDataLoader.ImageDataLoader
                train_data_loader = train_loader(
                    train_real_files,
                    train_fake_dataset_file_paths,
                    transform=transform,
                    max_frames_per_video=270,
                )
                train_data_loader.plot_image(idx=110, save_dir='Exp1/images/train')
            else:
                train_loader = dataset.data_loader.VideoFramesDataset.VideoFramesDataset
                train_data_loader = train_loader(
                    train_real_files,
                    train_fake_dataset_file_paths,
                    config,
                    train_dataset_names,
                    frames_per_clip=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH,
                    transform=transform,
                    max_frames_per_video=270,
                    rPPG_csv_path=rPPG_train_path
                )
                print("Train video dataloader len inside main.py, ", len(train_data_loader))
                train_data_loader.save_clip_plots(idx=1, save_dir='Exp1/chunks/train', frames_to_show=5)

            data_loader_dict['train'] = DataLoader(train_data_loader, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True, num_workers=8)
        else:
            data_loader_dict['train'] = None

        print("--- Loading validation data ---")
        if (not config.TEST.USE_LAST_EPOCH):
            if config.MODEL.NAME == "Xception":
                valid_loader = dataset.data_loader.ImageDataLoader.ImageDataLoader
                valid_dataset = valid_loader(
                    valid_real_files,
                    valid_fake_dataset_file_paths,
                    transform=transform,
                    max_frames_per_video=110
                )
                valid_dataset.plot_image(idx=110, save_dir='Exp1/images/valid')
            else:
                valid_loader = dataset.data_loader.VideoFramesDataset.VideoFramesDataset
                valid_dataset = valid_loader(
                    valid_real_files,
                    valid_fake_dataset_file_paths,
                    config,
                    valid_dataset_names,
                    frames_per_clip=config.VALID.DATA.PREPROCESS.CHUNK_LENGTH,
                    transform=transform,
                    max_frames_per_video=110,
                    rPPG_csv_path=rPPG_valid_path
                )
                valid_dataset.save_clip_plots(idx=1, save_dir='Exp1/chunks/valid', frames_to_show=5)
            
            data_loader_dict["valid"] = DataLoader(valid_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=8)
        else:
            data_loader_dict['valid'] = None

    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":

        test_real_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/original_sequences/youtube/c23/test.txt')


        # NOTE: for multiple ds
        test_dataset_names = config.TEST.DATA.DATASET.replace(" ", "").split(",")

        test_fake_dataset_file_paths = {}
        # NOTE: for multiple ds
        for test_dataset in test_dataset_names:
            test_fake_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/manipulated_sequences/{test_dataset}/c23/test.txt')
            test_fake_dataset_file_paths[test_dataset] = test_fake_files
            
        print("--- Loading testing data ---")
        if config.MODEL.NAME == "Xception":
            test_loader = dataset.data_loader.ImageDataLoader.ImageDataLoader
            test_data_loader = test_loader(
                    test_real_files,
                    test_fake_dataset_file_paths,
                    transform=transform,
                    max_frames_per_video=110
                )
            test_data_loader.plot_image(idx=110, save_dir='Exp1/images/test')
        else:
            test_loader = dataset.data_loader.VideoFramesDataset.VideoFramesDataset
            test_data_loader = test_loader(
                    test_real_files,
                    test_fake_dataset_file_paths,
                    config,
                    test_dataset_names,
                    frames_per_clip=config.TEST.DATA.PREPROCESS.CHUNK_LENGTH,
                    transform=transform,
                    max_frames_per_video=110,
                    rPPG_csv_path=rPPG_test_path

                )
            test_data_loader.save_clip_plots(idx=1, save_dir='Exp1/chunks/test', frames_to_show=5)
        data_loader_dict['test'] = DataLoader(test_data_loader, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=8)

    if config.TOOLBOX_MODE == "get_rPPG":

        train_real_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/original_sequences/youtube/c23/train.txt')
        valid_real_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/original_sequences/youtube/c23/val.txt')
        test_real_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/original_sequences/youtube/c23/test.txt')
        train_fake_dataset_file_paths={}
        valid_fake_dataset_file_paths={}
        test_fake_dataset_file_paths={}
        train_dataset_names=[]
        valid_dataset_names=[]
        test_dataset_names=[]

        train_loader = dataset.data_loader.VideoFramesDataset.VideoFramesDataset
        train_data_loader = train_loader(
            train_real_files,
            train_fake_dataset_file_paths,
            config,
            train_dataset_names,
            frames_per_clip=config.MODEL.PHYSNET.FRAME_NUM,
            transform=transform,
            max_frames_per_video=270
        )
        print("Train: save a clip plot")
        train_data_loader.save_clip_plots(idx=1, save_dir='Exp1/chunks/train/rPPG', frames_to_show=5)
        data_loader_dict['train'] = DataLoader(train_data_loader, batch_size=config.INFERENCE.BATCH_SIZE, shuffle=True, num_workers=8)
        print("Train dataloader length, ", len(train_data_loader))

        valid_loader = dataset.data_loader.VideoFramesDataset.VideoFramesDataset
        valid_dataset = valid_loader(
            valid_real_files,
            valid_fake_dataset_file_paths,
            config,
            valid_dataset_names,
            frames_per_clip=config.MODEL.PHYSNET.FRAME_NUM,
            transform=transform,
            max_frames_per_video=110
        )
        print("Valid: save a clip plot")
        valid_dataset.save_clip_plots(idx=1, save_dir='Exp1/chunks/valid/rPPG', frames_to_show=5)
        data_loader_dict["valid"] = DataLoader(valid_dataset, batch_size=config.INFERENCE.BATCH_SIZE, shuffle=False, num_workers=8)

        test_loader = dataset.data_loader.VideoFramesDataset.VideoFramesDataset
        test_data_loader = test_loader(
                test_real_files,
                test_fake_dataset_file_paths,
                config,
                test_dataset_names,
                frames_per_clip=config.MODEL.PHYSNET.FRAME_NUM,
                transform=transform,
                max_frames_per_video=110
            )
        test_data_loader.save_clip_plots(idx=1, save_dir='Exp1/chunks/test/rPPG', frames_to_show=5)
        data_loader_dict['test'] = DataLoader(test_data_loader, batch_size=config.INFERENCE.BATCH_SIZE, shuffle=False, num_workers=8)

    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "get_rPPG":
        get_rPPG(config, data_loader_dict)
    else:
        print("TOOLBOX_MODE only support train_and_test or only_test !", end='\n\n')
