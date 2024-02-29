""" The main function of rPPG deep learning pipeline."""

import argparse
import random
import time
from torchvision import transforms
import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from neural_methods.trainer.ResNet3DTrainer import ResNet3DTrainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader
import dataset.data_loader.VideoFramesDataset
from dataset.transforms import NormalizeVideo, ToTensorVideo, ToTensorVideoNoPermutation
from torch.utils.tensorboard import SummaryWriter


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


def train_and_test(config, data_loader_dict):
    # twriter = SummaryWriter(log_dir='Exp1/logs/NT_physnet_training_SGD_001')
    # vwriter = SummaryWriter(log_dir='Exp1/logs/NT_physnet_training_SGD_001')

    # iwriter = SummaryWriter(log_dir='Exp1/logs/videos_experiment')

    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
        twriter = SummaryWriter(log_dir=f'Exp1/logs/train/PhysNet/{config.TRAIN.MODEL_FILE_NAME}_training_SGD_{config.TRAIN.LR}_LR_reducer_full_dset')
        vwriter = SummaryWriter(log_dir=f'Exp1/logs/train/PhysNet/{config.TRAIN.MODEL_FILE_NAME}_training_SGD_{config.TRAIN.LR}_LR_reducer_full_dset')
    elif config.MODEL.NAME == "ResNet3D":
        model_trainer = ResNet3DTrainer(config, data_loader_dict)
        twriter = SummaryWriter(log_dir=f'Exp1/logs/train/ResNet3D/{config.TRAIN.MODEL_FILE_NAME}_training_SGD_{config.TRAIN.LR}_LR_reducer_full_dset')
        vwriter = SummaryWriter(log_dir=f'Exp1/logs/train/ResNet3D/{config.TRAIN.MODEL_FILE_NAME}_training_SGD_{config.TRAIN.LR}_LR_reducer_full_dset')
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
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
    model_trainer.train(data_loader_dict, twriter, vwriter)
    # test on same DS
    # model_trainer.test(data_loader_dict, twriter, vwriter)
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict):
    """Tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
        # twriter = SummaryWriter(log_dir=f'Exp1/logs/test/PhysNet/{config.TEST.DATA.DATASET}_PhysNet_test_full_dset')
        # vwriter = SummaryWriter(log_dir=f'Exp1/logs/test/PhysNet/{config.TEST.DATA.DATASET}_PhysNet_test_full_dset')
    elif config.MODEL.NAME == "ResNet3D":
        model_trainer = trainer.ResNet3D.ResNet3D(config, data_loader_dict)
        # twriter = SummaryWriter(log_dir=f'Exp1/logs/test/ResNet3D/{config.TEST.DATA.DATASET}_ResNet3D_test_full_dset')
        # vwriter = SummaryWriter(log_dir=f'Exp1/logs/test/ResNet3D/{config.TEST.DATA.DATASET}_ResNet3D_test_full_dset')
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
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
    # test on different ds
    # model_trainer.test(data_loader_dict, twriter, vwriter)
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
    if config.MODEL.NAME == "Physnet":
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

    if config.TOOLBOX_MODE == "train_and_test":
        # train_loader
        # if config.TRAIN.DATA.DATASET == "UBFC-rPPG":
        #     train_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        # elif config.TRAIN.DATA.DATASET == "PURE":
        #     train_loader = data_loader.PURELoader.PURELoader
        # elif config.TRAIN.DATA.DATASET == "SCAMPS":
        #     train_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        # elif config.TRAIN.DATA.DATASET == "MMPD":
        #     train_loader = data_loader.MMPDLoader.MMPDLoader
        # elif config.TRAIN.DATA.DATASET == "BP4DPlus":
        #     train_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        # elif config.TRAIN.DATA.DATASET == "BP4DPlusBigSmall":
        #     train_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        # elif config.TRAIN.DATA.DATASET == "UBFC-PHYS":
        #     train_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        # else:
        #     raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
        #                      SCAMPS, BP4D+ (Normal and BigSmall preprocessing), and UBFC-PHYS.")

        #NOTE: for one fake dataset
        # train_loader = dataset.data_loader.VideoFramesDataset.VideoFramesDataset
        # train_real_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/original_sequences/youtube/c23/train.txt')
        # train_fake_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/manipulated_sequences/{config.TRAIN.DATA.DATASET}/c23/train.txt')

        # valid_real_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/original_sequences/youtube/c23/val.txt')
        # valid_fake_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/manipulated_sequences/{config.VALID.DATA.DATASET}/c23/val.txt')

        # # Create and initialize the train dataloader given the correct toolbox mode,
        # # a supported dataset name, and a valid dataset paths
        # if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):

        #     train_data_loader = train_loader(
        #         train_real_files,
        #         train_fake_files,
        #         config,
        #         frames_per_clip=32,
        #         transform=transform,
        #     )
        #     data_loader_dict['train'] = DataLoader(train_data_loader, batch_size=6, shuffle=True, num_workers=8)
        #     train_data_loader.save_clip_plots(idx=12, save_dir='Exp1/chunks/train', frames_to_show=5)
        # else:
        #     data_loader_dict['train'] = None

        #NOTE: for multiple fake datasets
        train_loader = dataset.data_loader.VideoFramesDataset.VideoFramesDataset
        train_real_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/original_sequences/youtube/c23/train.txt')
        valid_real_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/original_sequences/youtube/c23/val.txt')
        
        train_dataset_names = config.TRAIN.DATA.DATASET.replace(" ", "").split(",")
        valid_dataset_names = config.VALID.DATA.DATASET.replace(" ", "").split(",")

        train_fake_dataset_file_paths = {}
        valid_fake_dataset_file_paths = {}

        for train_dataset in train_dataset_names:
            train_fake_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/manipulated_sequences/{train_dataset}/c23/train.txt')
            train_fake_dataset_file_paths[train_dataset] = train_fake_files
        
        for valid_dataset in valid_dataset_names:
            valid_fake_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/manipulated_sequences/{valid_dataset}/c23/val.txt')
            valid_fake_dataset_file_paths[valid_dataset] = valid_fake_files
            
        # Create and initialize the train dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset paths
        if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):

            train_data_loader = train_loader(
                train_real_files,
                train_fake_dataset_file_paths,
                config,
                train_dataset_names,
                frames_per_clip=32,
                transform=transform,
            )
            data_loader_dict['train'] = DataLoader(train_data_loader, batch_size=6, shuffle=True, num_workers=8)
            train_data_loader.save_clip_plots(idx=12, save_dir='Exp1/chunks/train', frames_to_show=5)
        else:
            data_loader_dict['train'] = None


        # valid_loader
        # if config.VALID.DATA.DATASET == "UBFC-rPPG":
        #     valid_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        # elif config.VALID.DATA.DATASET == "PURE":
        #     valid_loader = data_loader.PURELoader.PURELoader
        # elif config.VALID.DATA.DATASET == "SCAMPS":
        #     valid_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        # elif config.VALID.DATA.DATASET == "MMPD":
        #     valid_loader = data_loader.MMPDLoader.MMPDLoader
        # elif config.VALID.DATA.DATASET == "BP4DPlus":
        #     valid_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        # elif config.VALID.DATA.DATASET == "BP4DPlusBigSmall":
        #     valid_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        # elif config.VALID.DATA.DATASET == "UBFC-PHYS":
        #     valid_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        # elif config.VALID.DATA.DATASET is None and not config.TEST.USE_LAST_EPOCH:
        #     raise ValueError("Validation dataset not specified despite USE_LAST_EPOCH set to False!")
        # else:
        #     raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
        #                      SCAMPS, BP4D+ (Normal and BigSmall preprocessing), and UBFC-PHYS.")
        
        # Create and initialize the valid dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        # if (config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH):
            
        #NOTE: single dataset
        # valid_loader = dataset.data_loader.VideoFramesDataset.VideoFramesDataset

        # if (not config.TEST.USE_LAST_EPOCH):

        #     valid_dataset = valid_loader(
        #         valid_real_files,
        #         valid_fake_files,
        #         config,
        #         frames_per_clip=32,
        #         transform=transform,
        #     )
            
        #     data_loader_dict["valid"] = DataLoader(valid_dataset, batch_size=6, shuffle=False, num_workers=8)
        #     valid_dataset.save_clip_plots(idx=12, save_dir='Exp1/chunks/valid', frames_to_show=5)
        # else:
        #     data_loader_dict['valid'] = None
            
        #NOTE: multiple datasets
        valid_loader = dataset.data_loader.VideoFramesDataset.VideoFramesDataset

        if (not config.TEST.USE_LAST_EPOCH):

            valid_dataset = valid_loader(
                valid_real_files,
                valid_fake_dataset_file_paths,
                config,
                valid_dataset_names,
                frames_per_clip=32,
                transform=transform,
            )
            
            data_loader_dict["valid"] = DataLoader(valid_dataset, batch_size=6, shuffle=False, num_workers=8)
            valid_dataset.save_clip_plots(idx=12, save_dir='Exp1/chunks/valid', frames_to_show=5)
        else:
            data_loader_dict['valid'] = None
            

    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":

        test_real_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/original_sequences/youtube/c23/test.txt')
        # test_fake_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/manipulated_sequences/{config.TEST.DATA.DATASET}/c23/test.txt')
        # test_loader
        test_loader = dataset.data_loader.VideoFramesDataset.VideoFramesDataset

        # NOTE: for multiple ds
        test_dataset_names = config.TEST.DATA.DATASET.replace(" ", "").split(",")

        test_fake_dataset_file_paths = {}
        # NOTE: for multiple ds
        for test_dataset in test_dataset_names:
            test_fake_files = get_files_from_splits(f'/vol/research/DeepFakeDet/notebooks/FaceForensics++/manipulated_sequences/{test_dataset}/c23/val.txt')
            test_fake_dataset_file_paths[test_dataset] = test_fake_files

        test_data_loader = test_loader(
                test_real_files,
                test_fake_dataset_file_paths,
                config,
                test_dataset_names,
                frames_per_clip=32,
                transform=transform,
            )
        data_loader_dict['test'] = DataLoader(test_data_loader, batch_size=6, shuffle=False, num_workers=8)
        test_data_loader.save_clip_plots(idx=12, save_dir='Exp1/chunks/test', frames_to_show=5)


        # if config.TEST.DATA.DATASET == "UBFC-rPPG":
        #     test_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        # elif config.TEST.DATA.DATASET == "PURE":
        #     test_loader = data_loader.PURELoader.PURELoader
        # elif config.TEST.DATA.DATASET == "SCAMPS":
        #     test_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        # elif config.TEST.DATA.DATASET == "MMPD":
        #     test_loader = data_loader.MMPDLoader.MMPDLoader
        # elif config.TEST.DATA.DATASET == "BP4DPlus":
        #     test_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        # elif config.TEST.DATA.DATASET == "BP4DPlusBigSmall":
        #     test_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        # elif config.TEST.DATA.DATASET == "UBFC-PHYS":
        #     test_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        # else:
        #     raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
        #                      SCAMPS, BP4D+ (Normal and BigSmall preprocessing), and UBFC-PHYS.")
        
    #     if config.TOOLBOX_MODE == "train_and_test" and config.TEST.USE_LAST_EPOCH:
    #         print("Testing uses last epoch, validation dataset is not required.", end='\n\n')   

    #     # Create and initialize the test dataloader given the correct toolbox mode,
    #     # a supported dataset name, and a valid dataset path
    #     if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
    #         test_data = test_loader(
    #             name="test",
    #             data_path=config.TEST.DATA.DATA_PATH,
    #             config_data=config.TEST.DATA)
    #         data_loader_dict["test"] = DataLoader(
    #             dataset=test_data,
    #             num_workers=16,
    #             batch_size=config.INFERENCE.BATCH_SIZE,
    #             shuffle=False,
    #             worker_init_fn=seed_worker,
    #             generator=general_generator
    #         )
    #     else:
    #         data_loader_dict['test'] = None

    
    

    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test or only_test or unsupervised_method.")

    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    else:
        print("TOOLBOX_MODE only support train_and_test or only_test !", end='\n\n')
